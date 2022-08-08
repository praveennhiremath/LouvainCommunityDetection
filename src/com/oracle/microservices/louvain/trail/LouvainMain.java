package com.oracle.microservices.louvain.trail;

import java.util.concurrent.ExecutionException;

import oracle.pg.rdbms.AdbGraphClient;
import oracle.pg.rdbms.AdbGraphClientConfiguration;
import oracle.pgx.algorithm.PgxGraph;
import oracle.pgx.algorithm.VertexProperty;
import oracle.pgx.api.EdgeProperty;
import oracle.pgx.api.GraphSource;
import oracle.pgx.api.PgxSession;
import oracle.pgx.api.ServerInstance;
import oracle.pgx.common.types.PropertyType;

public class LouvainMain {

	public static void main(String[] args) throws ExecutionException, InterruptedException {
		
		Louvain louvain = new Louvain();
		
		//read from Cloud Graph 
		
		PgxGraph graph = null; EdgeProperty<Double> edgeProperty = null; int maxIter
		= 0; int nbrPass = 0; double total = 0; VertexProperty<Long> communityId =
		null;

		louvain.louvain(graph, edgeProperty, maxIter, nbrPass, total, communityId);
		
		
		
		
		 
		var config = AdbGraphClientConfiguration.builder();
		config.tenant("Your Tenancy");
		config.database("Your DB");
		config.username("username");
		config.password("password");
		config.endpoint("https://bsenjiat5lmurtq-medicalrecords.adb.us-ashburn-1.oraclecloudapps.com/");
		
		var client = new AdbGraphClient(config.build());
		
		
		ServerInstance instance = client.getPgxInstance();
		PgxSession session = instance.createSession("MedicalRecAdbGraphSessionName");
		
		//PgxGraph graph = (PgxGraph) session.readGraphByName("MEDICAL_RECS_G", GraphSource.PG_VIEW);
		
		PgxGraph graph = (PgxGraph) session.readGraphByName("MEDICAL_RECS_G", GraphSource.PG_VIEW);
		
		oracle.pgx.algorithm.PgxGraph graphAPI = (PgxGraph) session.readGraphByName("MEDICAL_RECS_G", GraphSource.PG_VIEW);
		
		EdgeProperty<Double> weightProp = graphAPI.getEdgeProperty("TOTAL_AFFINITY");

		//session.createAnalyst().louvain(graph, weightProp)
		//VertexProperty<Long> communityId = graph.getVertexProperty("PropertyType.LONG");
		
		VertexProperty<Long> communityId = VertexProperty.create();
		
		VertexProperty property = graph.createVertexProperty(PropertyType.INTEGER);
		
		//session.createAnalyst().louvain(graph, weightProp);
		
		System.out.println("COmpleted");
		double doub = 0;
		louvain.louvain(graphAPI, weightProp,0,0,doub,communityId);
		
	  }
}
