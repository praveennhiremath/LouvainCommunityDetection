package com.oracle.microservices.louvain;

import java.util.concurrent.ExecutionException;

import oracle.pg.rdbms.AdbGraphClient;
import oracle.pg.rdbms.AdbGraphClientConfiguration;
import oracle.pgx.api.PgxGraph;
import oracle.pgx.api.CompiledProgram;
import oracle.pgx.api.PgxSession;
import oracle.pgx.api.ServerInstance;
import oracle.pgx.api.VertexProperty;
import oracle.pgx.common.types.PropertyType;

public class CompliledLouvain {

	public static void main(String[] args) throws ExecutionException, InterruptedException {
		
		var config = AdbGraphClientConfiguration.builder();
		config.tenant("Your Tenancy");
		config.database("Your DB");
		config.username("username");
		config.password("password");
		config.endpoint("https://bsenjiat5lmurtq-medicalrecords.adb.us-ashburn-1.oraclecloudapps.com/");
		
		var client = new AdbGraphClient(config.build());
		
		ServerInstance instance = client.getPgxInstance();
		PgxSession session = instance.createSession("MedicalRecAdbGraphSessionName");
		
		//MEDICAL_RECS_G
		PgxGraph graph = session.readGraphByName("MEDICAL_RECORDS_GRAPH", oracle.pgx.api.GraphSource.PG_VIEW);
		CompiledProgram myAlgorithm = session.compileProgram(".\\Louvain.java");
		VertexProperty property = graph.createVertexProperty(PropertyType.INTEGER);
		
		myAlgorithm.run(graph, property);
		
	}
}
