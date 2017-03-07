/* -*- mode: C; mode: folding; fill-column: 70; -*- */
#define _XOPEN_SOURCE 600
#define _LARGEFILE64_SOURCE 1
#define _FILE_OFFSET_BITS 64
#undef _OPENMP

// #if defined(_OPENMP)
// #include "omp.h"
// #endif

#include "stinger_core/stinger_atomics.h"
#include "stinger_utils/stinger_utils.h"
#include "stinger_core/stinger.h"
#include "stinger_utils/timer.h"
#include "stinger_core/xmalloc.h"

#define ACTI(k) (action[2*(k)])
#define ACTJ(k) (action[2*(k)+1])

static int64_t nv, ne, naction;
static int64_t * restrict off;
static int64_t * restrict from;
static int64_t * restrict ind;
static int64_t * restrict weight;
static int64_t * restrict action;

/* handles for I/O memory */
static int64_t * restrict graphmem;

static char * initial_graph_name = INITIAL_GRAPH_NAME_DEFAULT;
static char * action_stream_name = ACTION_STREAM_NAME_DEFAULT;

static long batch_size = BATCH_SIZE_DEFAULT;
static long nbatch = NBATCH_DEFAULT;

static struct stinger * S;

static double * delete_time_trace;
static double * insert_time_trace;

typedef int64_t length_t;
typedef int64_t vertexId_t;

int getline(char **line, size_t *n, FILE* fp) {
  *line = malloc(1024);
  char* ret = fgets(*line, 1024, fp);
  if(!ret) {
    *n = -1;
    return -1;
  }
  *n = strlen(ret);
  return *n;
}

void readGraphDIMACS(char* filePath, int64_t** prmoff, int64_t** prmind,
                     int64_t* prmnv, int64_t* prmne, int isRmat)
{
    printf("Filename: %s\n",filePath);
    FILE *fp = fopen (filePath, "r");
    char* line = NULL;

    // Read data from file
    int32_t temp, lineRead;
    size_t bytesRead = 0;
    getline (&line, &bytesRead, fp);

    sscanf (line, "%d %d", &nv, &ne);   

    free(line);

    off = (length_t *) malloc ((nv + 2) * sizeof (length_t)); 
    nv++;
    if(!isRmat){
        ind = (vertexId_t *) malloc ((ne * 2) * sizeof (vertexId_t));        
        ne *= 2;        
    }
    else{
        ind = (vertexId_t *) malloc ((ne) * sizeof (vertexId_t));        
    }
    

    off[0] = 0;
    off[1] = 0;
    length_t counter = 0;
    vertexId_t u;
    line = NULL;
    bytesRead = 0;

    for (u = 1; (temp = getline (&line, &bytesRead, fp)) != -1; u++)
    {
        vertexId_t neigh = 0;
        vertexId_t v = 0;
        char *ptr = line;
        int read = 0;
        char tempStr[1000];
        lineRead = 0;
        while (lineRead < bytesRead && (read = sscanf (ptr, "%s", tempStr)) > 0)
        {
            v = atoi(tempStr);
            read = strlen(tempStr);
            ptr += read + 1;
            lineRead = read + 1;
            neigh++;
            ind[counter++] = v;
        }
        off[u + 1] = off[u] + neigh;
        free(line);
        bytesRead = 0;
    }
    fclose (fp);

    *prmnv = nv;
    *prmne = ne;
    *prmind = ind;
    *prmoff = off;
}

struct stinger* load_csr(char * name){
  int64_t *sv, *ev, *w;
  readGraphDIMACS(name, &off, &ind, &nv, &ne, 0);
  sv = malloc(ne*sizeof(int64_t));
  ev = malloc(ne*sizeof(int64_t));
  w = malloc(ne*sizeof(int64_t));
  memset(w,0,ne*sizeof(int64_t));

  int i = 0;
	for (int v=0; v < nv; v++){
		for(int32_t e=0; e<off[v+1]-off[v]; e++){
      sv[i] = v;
      ev[i] = ind[off[v]+e];
      i++;
		}
	}

  struct stinger* ret = edge_list_to_stinger	(nv, ne, sv, ev, w, 0, 0, 0);
  free(off);
  free(ind);
  free(sv);
  free(ev);
  free(w);
  return ret;
}

void random_edges(int n) {
  for(uint64_t k = 0; k < n; k++) {
    const int64_t i = rand() % nv;
    const int64_t j = rand() % nv;
    ACTI(k) = i;
    ACTJ(k) = j;
  }
}

int existing_edges(int n) {
  int64_t k;
  k = 0;
  STINGER_READ_ONLY_FORALL_EDGES_BEGIN(S, 0) {
    if(k < n) {
      ACTI(k) = STINGER_RO_EDGE_SOURCE;
      ACTJ(k) = STINGER_RO_EDGE_DEST;
    } else {
      return k;
    }
    k++;
  } STINGER_READ_ONLY_FORALL_EDGES_END();
}

int
main (const int argc, char *argv[])
{
  parse_args (argc, argv, &initial_graph_name, &action_stream_name, &batch_size, &nbatch);
  STATS_INIT();

  S = load_csr(initial_graph_name);
  for(batch_size = 1000; batch_size < ne; batch_size*=2) {
    nbatch = 10;
    naction = batch_size*nbatch;
    action = (int64_t*) malloc(2*naction*sizeof(int64_t));
    print_initial_graph_stats (nv, ne, batch_size, nbatch, naction);
    fflush(stdout);
    BATCH_SIZE_CHECK();

#if defined(_OPENMP)
    /* OMP("omp parallel") */
    {
      /* OMP("omp master") */
      PRINT_STAT_INT64 ("num_threads", (long int) omp_get_num_threads());
    }
#endif

    insert_time_trace = xmalloc (nbatch * sizeof(*insert_time_trace));
    delete_time_trace = xmalloc (nbatch * sizeof(*delete_time_trace));

    /* Convert to STINGER */
    tic ();
    uint32_t errorCode = stinger_consistency_check (S, nv);
    double time_check = toc ();
    /* PRINT_STAT_HEX64 ("error_code", (long unsigned) errorCode); */
    /* PRINT_STAT_DOUBLE ("time_check", time_check); */

    /* Updates */
    int64_t ntrace = 0;
    srand(batch_size);
    for (int64_t actno = 0; actno < nbatch * batch_size; actno += batch_size)
      {

        const int64_t endact = (actno + batch_size > naction ? naction : actno + batch_size);
        int64_t numActions = endact - actno;
        existing_edges(numActions);

        tic();
        /* MTA("mta assert parallel") */
        /* MTA("mta block dynamic schedule") */
        /* OMP("omp parallel for") */
        for(uint64_t k = 0; k < numActions; k++) {
          stinger_remove_edge(S, 0, ACTI(k), ACTJ(k));
        }

        insert_time_trace[ntrace] = toc();
        ntrace++;

      } /* End of batch */

    ntrace = 0;
    srand(batch_size);
    for (int64_t actno = 0; actno < nbatch * batch_size; actno += batch_size)
      {

        const int64_t endact = (actno + batch_size > naction ? naction : actno + batch_size);
        int64_t numActions = endact - actno;
        existing_edges(numActions);

        /* MTA("mta assert parallel") */
        /*   MTA("mta block dynamic schedule") */
        /*   OMP("omp parallel for") */
        /* random_edges(numActions); */
        /* existing_edges(numActions); */

        tic();
        for(uint64_t k = 0; k < endact - actno; k++) {
          stinger_insert_edge (S, 0, ACTI(k), ACTJ(k), 1, actno+2);
        }

        delete_time_trace[ntrace] = toc();
        ntrace++;

      } /* End of batch */

    /* Print the times */
    double time_insert = 0;
    for (int64_t k = 0; k < nbatch; k++) {
      time_insert += insert_time_trace[k];
    }
    PRINT_STAT_DOUBLE ("time_insert", time_insert/nbatch);
    /* PRINT_STAT_DOUBLE ("insert_per_sec", (nbatch * batch_size) / time_insert);  */

    double time_delete = 0;
    for (int64_t k = 0; k < nbatch; k++) {
      time_delete += delete_time_trace[k];
    }
    PRINT_STAT_DOUBLE ("time_delete", time_delete/nbatch);
    /* PRINT_STAT_DOUBLE ("delete_per_sec", (nbatch * batch_size) / time_delete);  */

    tic ();
    errorCode = stinger_consistency_check (S, nv);
    time_check = toc ();
    fflush(stdout);
    /* PRINT_STAT_HEX64 ("error_code", (long unsigned) errorCode); */
    /* PRINT_STAT_DOUBLE ("time_check", time_check); */

    free(insert_time_trace);
    free(delete_time_trace);
    free(action);
  }
  stinger_free_all (S);
  STATS_END();
  fflush(stdout);
}
