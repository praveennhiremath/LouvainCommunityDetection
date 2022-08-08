package com.oracle.microservices.louvain.trail;

import java.util.concurrent.ExecutionException;

import oracle.pg.rdbms.*;
import oracle.pgx.algorithm.EdgeProperty;
import oracle.pgx.algorithm.PgxGraph;
import oracle.pgx.algorithm.VertexProperty;
import oracle.pgx.api.Analyst;
/*import oracle.pgx.api.Analyst;
import oracle.pgx.api.Pgx;
import oracle.pgx.api.PgxSession;
import oracle.pgx.api.ServerInstance;*/
import oracle.pgx.algorithm.*;

/*import oracle.pg.rdbms.AdbGraphClient;
import oracle.pg.rdbms.AdbGraphClientConfiguration;
import oracle.pgx.algorithm.PgxGraph;
import oracle.pgx.algorithm.VertexProperty;
import oracle.pgx.api.Pgx;
import oracle.pgx.api.PgxSession;
import oracle.pgx.api.ServerInstance;
import oracle.pgx.algorithm.EdgeProperty;
import oracle.pgx.common.types.PropertyType;
import oracle.pgx.config.GraphConfig;*/

public class LouvainMainAlgoPackage {

	public static void main(String[] args) throws ExecutionException, InterruptedException {
		
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
		
		

		PgxSession session1 = Pgx.createSession("my-session");
		
		//PgxSession session1 = Pgx.createSession("MedicalRecAdbGraphSessionName");
		
		//PgxGraph graph = (PgxGraph) session.readGraphByName("MEDICAL_RECS_G", GraphSource.PG_VIEW);
		
		// GraphConfig.
		
		PgxGraph graph = session.readGraphByName("MEDICAL_RECS_G", oracle.pgx.api.GraphSource.PG_VIEW);
		//oracle.pgx.api.PgxGraph graph1111 = session.readGraphByName("MEDICAL_RECS_G", oracle.pgx.api.GraphSource.PG_VIEW);
		PgxGraph graph11 = session.readGraphWithProperties("/path/to/config.edge.json");
		
		Analyst analyst = session.createAnalyst();
		VertexProperty<Integer, Double> rank = analyst.louvain(null, null);
		
		// accessing build-in class
		/*EdgeProperty<Double> weightProp = EdgeProperty.create();
		oracle.pgx.api.EdgeProperty<Double> edgeProp = graph1111.getEdgeProperty("TOTAL_AFFINITY");*/

		//session.createAnalyst().louvain(graph, weightProp)
		//VertexProperty<Long> communityId = graph.getVertexProperty("PropertyType.LONG");
		
		VertexProperty<Long> communityId = VertexProperty.create();
		
		session.createAnalyst().louvain(graph11, edgeProp);
		
		System.out.println("Completed!!!");
		
		double doub = 0;
		louvain.louvain(graph, weightProp, 0, 0, doub, communityId);
		
		/*
		 * PgqlResultSet rs =
		 * graph.queryPgql("Pass PGX query for accessing MEDICAL_REC_G here to test");
		 * rs.print();
		 */
		
		try (Analyst analyst1 = session.createAnalyst()) {
			  analyst.louvain(graph, weightProp, 10, 10, doub, communityId);
			}
		
		

		
	  }
}
