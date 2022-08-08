package com.oracle.microservices.louvain;

import static java.lang.Math.pow;

import oracle.pg.rdbms.AdbGraphClient;
import oracle.pg.rdbms.AdbGraphClientConfiguration;
import oracle.pgx.algorithm.EdgeProperty;
import oracle.pgx.algorithm.PgxEdge;
import oracle.pgx.algorithm.PgxGraph;
import oracle.pgx.algorithm.PgxMap;
import oracle.pgx.algorithm.VertexProperty;
import oracle.pgx.algorithm.VertexSet;
import oracle.pgx.algorithm.annotations.GraphAlgorithm;
import oracle.pgx.algorithm.annotations.Out;
import oracle.pgx.api.Analyst;
import oracle.pgx.api.Pgx;
import oracle.pgx.api.PgxSession;
import oracle.pgx.api.ServerInstance;


@GraphAlgorithm
public class Louvain {
	
	long c ;
	double kIn;
    double gain;
    double kInOld;
    double kInNew;
    double maxGain;
    double modularityGain;
    boolean changed;
    double q;
    
  public void louvain(PgxGraph g, EdgeProperty<Double> weight, int maxIter, int nbrPass, double tol,
      @Out oracle.pgx.algorithm.VertexProperty<Long> communityId) {
	  
	  System.out.println("------------- In Louvain -------------");
    c=0;
    int iter = 0;
    int pass = 0;
    long numVertices = g.getNumVertices();
    PgxMap<Long, Double> sumIn = PgxMap.create();
    PgxMap<Long, Double> sumTotal = PgxMap.create();
    oracle.pgx.algorithm.VertexProperty<Double> edgeWeight = null;
    // initialize communities
    g.getVertices().forSequential(n -> {
    	
      communityId.set(n, c);

      //sum of the weights of the edges incident to node n
      edgeWeight.set(n, n.getOutEdges().sum(weight));

      //sum of the weights of the edges inside n's community
      sumIn.set(c, n.getOutEdges().filter(e -> e.destinationVertex() == n).sum(weight));

      //sum of the weights of the edges incident to nodes in n's community
      sumTotal.set(c, edgeWeight.get(n));

      c++;
    });

    double twoM = 2 * g.getEdges().sum(weight);
    double newMod = 0;
    double curMod = 0;

    if (nbrPass > 1) {
      newMod = modularity(g, weight, edgeWeight, communityId, twoM);
    }

  //  boolean changed;

    do {
      curMod = newMod;

      //aggregate the graph: nodes of the same community constitute a super node
      PgxMap<Long, VertexSet> svertices = PgxMap.create();
      PgxMap<Long, VertexSet> superNbrs = PgxMap.create();
      PgxMap<Long, Double> allSuperEdges = PgxMap.create();
      VertexProperty<Long> svertex = VertexProperty.create();
      PgxMap<Long, Long> svertexCommunity = PgxMap.create();
      PgxMap<Long, Double> edgeWeightSum = PgxMap.create();

      g.getVertices().forSequential(n -> {
        svertices.get(communityId.get(n)).add(n);
        svertexCommunity.set(communityId.get(n), communityId.get(n));
        svertex.set(n, communityId.get(n));
        n.getOutNeighbors().forSequential(nNbr -> {
          PgxEdge e = nNbr.edge();
          long idx = (numVertices * communityId.get(n)) + communityId.get(nNbr);
          if (!allSuperEdges.containsKey(idx)) {
            superNbrs.get(communityId.get(n)).add(nNbr);
          }
          allSuperEdges.reduceAdd(idx, weight.get(e));
          edgeWeightSum.reduceAdd(communityId.get(n), weight.get(e));
        });
      });

      do {
        changed = false;
        svertices.keys().forSequential(n -> {
          c = svertexCommunity.get(n);
          kIn = 0;
          gain = 0;
          VertexSet snbrs = superNbrs.get(n).clone();
          maxGain = 0;
          modularityGain = 0;
          snbrs.forSequential(o -> {
            Long comm = svertexCommunity.get(svertex.get(o));
            snbrs.forSequential(m -> {
              if (svertexCommunity.get(svertex.get(m)) == comm) {
            	  kIn += allSuperEdges.get((numVertices * n) + svertex.get(m));
              }
            });
            modularityGain = (sumIn.get(comm) + kIn) / twoM - pow((sumTotal.get(comm) + edgeWeightSum.get(n)) / twoM,
                    2) - (sumIn.get(comm) / twoM - pow(sumTotal.get(comm) / twoM, 2) - pow(edgeWeightSum.get(n) / twoM, 2));
            
            if (modularityGain > maxGain) {
              maxGain = modularityGain;
              svertexCommunity.set(n, svertexCommunity.get(svertex.get(o)));
            }
          });

          if (svertexCommunity.get(n) != c) {
            kInOld = 0;
            kInNew = 0;
            changed = true;
            
            snbrs.forSequential(m -> {
              if (svertexCommunity.get(svertex.get(m)) == c) {
            	  kInOld += allSuperEdges.get((numVertices * n) + svertex.get(m));
              }
            });
            sumIn.set(c, sumIn.get(c) - kInOld);
            sumTotal.set(c, sumTotal.get(c) - (edgeWeightSum.get(n) - kInOld));
            snbrs.forSequential(m -> {
              if (svertexCommunity.get(svertex.get(m)) == svertexCommunity.get(n)) {
                kInNew += allSuperEdges.get((numVertices * n) + svertex.get(m));
              }
            });
            sumIn.set(svertexCommunity.get(n), sumIn.get(svertexCommunity.get(n)) + kInNew);
            sumTotal.set(svertexCommunity.get(n), sumTotal.get(svertexCommunity.get(n))
                + (edgeWeightSum.get(n) - kInNew));
          }
        });
        iter++;
      } while (changed && iter < maxIter);
      g.getVertices().forEach(n -> {
        communityId.set(n, svertexCommunity.get(svertex.get(n)));
      });
      pass++;
      if (nbrPass > 1) {
        newMod = modularity(g, weight, edgeWeight, communityId, twoM);
      }
    } while (pass < nbrPass && (newMod - curMod > tol));
  }

  double modularity(PgxGraph g, EdgeProperty<Double> weight, VertexProperty<Double> edgeWeight,
      VertexProperty<Long> communityId, double twoM) {
    q = 0;
    g.getVertices().forEach(i -> {
      g.getVertices().forEach(j -> {
        if (communityId.get(i) == communityId.get(j)) {
          double aij = j.getOutEdges().filter(e -> e.destinationVertex() == i).sum(weight);
          q += aij - (edgeWeight.get(i) * edgeWeight.get(j) / twoM);
        }
      });
    });
    return q / twoM;
  }
 
  public static void main(String[] args) {
	  Louvain louvain = new Louvain();
		
		//read from Cloud Graph 
		/*
		 * PgxGraph graph = null; EdgeProperty<Double> edgeProperty = null; int maxIter
		 * = 0; int nbrPass = 0; double total = 0; VertexProperty<Long> communityId =
		 * null;
		 * 
		 * louvain.louvain(graph, edgeProperty, maxIter, nbrPass, total, communityId);
		 */
		 
		var config = AdbGraphClientConfiguration.builder();
		config.tenant("ocid1.tenancy.oc1..aaaaaaaabls4dzottlktt774tu3knax6crpozycjhrqpm73thfryxwlkmkba");
		config.database("medicalrecordsdb");
		config.username("GRAPHUSER");
		config.password("Welcome12345");
		config.endpoint("https://bsenjiat5lmurtq-medicalrecordsdb.adb.us-ashburn-1.oraclecloudapps.com/");
		
		
		var client = new AdbGraphClient(config.build());
		
		ServerInstance instance = client.getPgxInstance();
		PgxSession session = instance.createSession("MedicalRecAdbGraphSessionName");
		//PgxSession session1 = Pgx.createSession("my-session");
		
		oracle.pgx.api.PgxGraph graph = session.readGraphByName("MEDICAL_RECS_G", oracle.pgx.api.GraphSource.PG_VIEW);
		
		//oracle.pgx.api.PgxGraph graph1111 = session.readGraphByName("MEDICAL_RECS_G", oracle.pgx.api.GraphSource.PG_VIEW);
		//PgxGraph graph11 = session.readGraphWithProperties("/path/to/config.edge.json");
		
		Analyst analyst = session.createAnalyst();
		
		// accessing build-in class
		EdgeProperty<Double> weightProp = EdgeProperty.create();
		/*oracle.pgx.api.EdgeProperty<Double> edgeProp = graph1111.getEdgeProperty("TOTAL_AFFINITY");*/
		//session.createAnalyst().louvain(graph, weightProp)
		
		//VertexProperty<Long> communityId = graph.getVertexProperty("PropertyType.LONG");
		VertexProperty<Long> communityId = VertexProperty.create();
		
		
		System.out.println("Completed!!!");
		
		double doub = 0;
		louvain.louvain(graph, weightProp, 0, 0, doub, communityId);
		
		try (Analyst analyst1 = session.createAnalyst()) {
			analyst1.louvain(graph, weightProp, 10, 10, doub, communityId);
		}
	  }
  
}