
 Copyright (c) 2011, 2022, Oracle and/or its affiliates. All rights reserved.

   NAME
     README.txt - Oracle Graph Server and Client Release Information

     Release 22.1.0 README file


  Contents
  ========
    Supported Software Versions
    What's new on Oracle Graph Server and Client 22.1.0
    What's new on Oracle Graph Server and Client 21.4.2
    What's new on Oracle Graph Server and Client 20.4.5
    What's new on Oracle Graph Server and Client 21.4.1
    What's new on Oracle Graph Server and Client 21.4.0
    What's new on Oracle Graph Server and Client 20.4.4
    What's new on Oracle Graph Server and Client 21.3.0
    What's new on Oracle Graph Server and Client 20.4.3
    What's new on Oracle Graph Server and Client 21.2.0
    What's new on Oracle Graph Server and Client 20.4.2
    What's new on Oracle Graph Server and Client 21.1.0
    What's new on Oracle Graph Server and Client 20.4.1
    What's new on Oracle Graph Server and Client 20.4.0
    What's new on Oracle Graph Server and Client 20.3.0
    What's new on Oracle Graph Server and Client 20.2.0
    What's new on Oracle Graph Server and Client 20.1.0

  Supported Software Versions
  ===========================

  Oracle Graph Server and Client supports the following database distributions and versions:
  - Oracle Database 12.2
  - Oracle Database 19c
  - Oracle Database 21c
  - Oracle Autonomous Transaction Processing
  - Oracle Autonomous Data Warehouse

  Some versions are supported with limitations. Refer to this blog post
  for a detailed compatibility matrix:
  https://blogs.oracle.com/oraclespatial/database-compatibility-matrix-for-oracle-graph-server-and-client

  Oracle Graph Web Applications can be deployed into the following external web application servers
  - Oracle WebLogic Server 12.2.x
  - Oracle WebLogic Server 14c
  - Apache Tomcat 9.0.x

  Oracle Graph Server and Client supports the following data stores distributed
  by Cloudera:
  - Apache Hadoop Distributed Filesystem (HDFS) of Cloudera CDH 6.x
  - Apache Hadoop Distributed Filesystem (HDFS) of Cloudera CDH 7.x

  Connecting to HDFS requires an additional installation of the Oracle Graph HDFS connectors
  packages. Please refer to the README of the Apache Hadoop connector package for instructions.

  What's new on Oracle Graph Server and Client 22.1.0
  ===================================================

  Main new features on Oracle Graph Server and Client 22.1.0:

   - Added support for reading tables of a partitioned graph or PG view at a given SCN
   - Added better support for running Graph Server behind a load balancer
   - Added support for path unnesting in PGQL on PGX
   - Added support for custom property names and new OPTIONS in CREATE PROPERTY GRAPH in PGQL on PGX
   - Added support for selecting all properties through SELECT x.* in both PGQL on PGX and PGQL on PG Views
   - PGQL on PG Views improvements
      - Added support for more quantifiers besides '*'
      - Added support for precision and scale in CAST
   - Added support to define Oracle RDF datasource to be public for web users with proper privileges

  The following problems were fixed in Oracle Graph Server and Client 22.1.0:

    - CVE-2021-44228, CVE-2021-44832, CVE-2021-45046 and CVE-2021-45105 in 3rd party component log4j
    - BUG 33420248: vertexBetweennessCentrality() fails with a "not implemented" exception
    - BUG 32886775: error message not helpful during login if DB is unreachable
    - BUG 33048064: PG_VIEW: can't access result from COUNT as long or integer or string
    - BUG 33171116: PG_VIEW: incorrect "Property does not exist" error
    - BUG 33204828: readGraphFiles API throws ClassCastException
    - BUG 33313502: PgqlToSqlException: ORA-01789: query block has incorrect number of result columns
    - BUG 33360416: PG_VIEW: ORA-00923: FROM keyword not found where expected
    - BUG 33417681: Graphviz generating incorrect PGQL with subqueries
    - BUG 33450490: invalid PGQL syntax causes NullPointerException
    - BUG 33570489: starting OPG4PY Shell - errors if username not provided
    - BUG 33606500: opg4j and opg4py does not show error message if kerberos ticket is not valid
    - BUG 33659349: null values from SPARQL query not being added in CSV and TSV stream formats
    - BUG 33617302: GraphViz throws an error when label is specified and graph contains too many properties
    - BUG 33719548: PG_VIEW: graph names not quoted

  What's new on Oracle Graph Server and Client 21.4.2
  ===================================================

  This is a patch release on top of version 21.4.0 which contains fixes for the following issues:

    - GNNExplainer did not properly check the user permission for new graph creation
    - User created ids for vertices did not work for GNNExplainer
    - Unsupervised Graphwise inferAndGetExplanation(...) could not run on the partitioned graph
    - DeepWalk and GraphWise models could not run on graphs with partitioned ids
    - CVE-2021-44228 and CVE-2021-45046 in 3rd party component log4j

  What's new on Oracle Graph Server and Client 20.4.5
  ===================================================

  This is the fifth limited patch cycle release on top of version 20.4.0
  which contains fixes for the following issues:

    - CVE-2021-44228 and CVE-2021-45046 in 3rd party component log4j

  What's new on Oracle Graph Server and Client 21.4.1
  ===================================================

  This is a patch release on top of version 21.4.0 which contains fixes for the following issues:

    - Bug 33539457: PGQL on RDBMS queries fail with ORA-942 when locale is other than English

  What's new on Oracle Graph Server and Client 21.4.0
  ===================================================

  Main new features on Oracle Graph Server and Client 21.4.0:

   - Added better PGQL Type checking during query parsing
      - Check for existence of labels and properties
      - Verify consistency of data types for several types of operations
      - Give more meaningful error messages that include the exact location of the error in the query text
      - Provide consistent error message across different PGQL implementations (PGQL on RDBMS only for PG Views)
   - Added support for Anomaly Detection to GraphWise using Deviation Network technique
   - Added support for Graph Alteration without requiring data source
   - Added support for UDF (User Defined Functions) and Oracle Database functions for PGQL on PG Views
   - Added support for sticky sessions during server and client communications

  The following problems were fixed in Oracle Graph Server and Client 21.4.0:

    - BUG 33070670: values of properties in GraphViz are mixed up
    - BUG 32702716: store query enhancement on session/cache
    - BUG 33070670: GraphViz shows wrong properties in loaded graph
    - BUG 33088929: no properties are returned when using PG Views
    - BUG 33161882: Python iterator not working in opg4py
    - BUG 33195641: Unable to grant file permissions using Autonomous Database
    - BUG 33205170: HTTPS communication asking for alternate name in the certificate
    - BUG 33225801: GraphViz throws 404 if jdbc_url on web.xml is malformed
    - BUG 33332400: GraphViz throws NullPointerException on WebLogic Server

  What's new on Oracle Graph Server and Client 20.4.4
  ===================================================

  This is the fourth limited patch cycle release on top of version 20.4.0
  which contains fixes for the following issues:

    - CVE-2021-30639 in 3rd party component Apache Tomcat
    - CVE-2021-35515 in 3rd party component Commons-compress

  What's new on Oracle Graph Server and Client 21.3.0
  ===================================================

  Main new features on Oracle Graph Server and Client 21.3.0:

   - Added support for PGQL 1.4 (see https://pgql-lang.org/spec/1.4/)
      - Added new path finding goals: ANY, ALL SHORTEST and ALL paths
      - New string operators and functions: concatenation (||), LISTAGG, UPPER, LOWER, SUBSTRING
      - Note: Graph Server and Client 21.3.0 is the first reference implementation of PGQL 1.4
        and fully backwards compatible with older versions of PGQL (1.0, 1.1, 1.2 and 1.3)
      - Note: there are known limitations between PGQL implementations on PGX, PG schema and PG views.
        Please consult the user manual for a detailed list.
   - Added APIs to check user and graph permissions
   - Added an API to read PG views by name
   - Allow programmatic creation of PGX frames and partitioned graphs from frames
   - Added Python API for PGQL on Oracle Database
   - Added support to explain a GNN model from PGX (e.g. GraphWise)
   - Graph Visualization Enhancements:
      - Added support for Oracle Database Kerberos authentication
      - Allow configuration of the PGQL driver (PGX or Database) during login

  The following problems were fixed in Oracle Graph Server and Client 21.3.0:

    - BUG 32425444: GraphViz returns inconsistent results depending on the properties being queried
    - BUG 32780171: memory cleanup log messages pollute log if graph server runs in DEBUG mode
    - BUG 32809315: storing frame without DB connection info throws NullPointerException
    - BUG 32800163: Vertex/Edge labels in GraphViz settings not in alphabetical order
    - BUG 32804567: validity of Kerberos cache directory checked even if Kerberos disabled
    - BUG 32804568: wrong error message if configured Kerberos cache directory does not exist
    - BUG 32817180: PGQL on Oracle Database: INSERT in combination with GROUP BY not working
    - BUG 33069191: RDF server: blank contents on model page in Tab component

  What's new on Oracle Graph Server and Client 20.4.3
  ===================================================

  This is the third limited patch cycle release on top of version 20.4.0
  which contains fixes for the following issues:

    - CVE-2021-25122 in 3rd party component Apache Tomcat
    - CVE-2021-23337 in 3rd party component lodash

  What's new on Oracle Graph Server and Client 21.2.0
  ===================================================

  Main new features on Oracle Graph Server and Client 21.2.0:

    - New graph machine learning algorithms
      -- Deep Graph Infomax
      -- GraphWise with edge property
    - Added new "Partitioned" ID generation strategy to PGX. With this new strategy, reading from partitioned tables
      no longer requires the tables to have graph-globally unique primary keys.
    - Extended shortest path query support for PGQL on PGX
      -- TOP K SHORTEST
      -- ALL SHORTEST
      -- min/max hop constraints
    - JDBC Driver for PGQL on RDBMS
    - Graph Visualization Web Application: added support for query cancellation
    - Python client: added to_pandas() function to convert PGQL result set to Pandas data frame
    - Support for Oracle Database configured with Kerberos authentication

  The following problems were fixed in Oracle Graph Server and Client 21.2.0:

    - Bug 31921010 - PGQL prompt stays after exiting PGQL mode in PGQL plugin for SQLcl
    - Bug 32080347 - PGQL on RDBMS UPDATE statement fails if parallelism is set to more than 1
    - Bug 32394252 - Graph Visualization application throws error when querying PG views
    - Bug 32312335 - datasource_dir_whitelist config parameter should be removed from pgx.conf
    - Bug 32568603 - Confusing error when attempting to access a pre-loaded graph without permission
    - Bug 32593962 - SPARQL CONSTRUCT query not being recognized in RDF Query UI web application
    - Bug 32663841 - Python client is not refreshing authentication token in interactive remote mode

  What's new on Oracle Graph Server and Client 20.4.2
  ===================================================

  This is the second limited patch cycle release on top of version 20.4.0
  which contains fixes for the following issues:

    - Bug 32684076 - fixed "No suitable driver found" error in Tomcat deployments when logging in
    - CVE-2020-8908 in 3rd party component Google Guava
    - CVE-2020-13956 in 3rd party component Apache HttpClient
    - CVE-2020-17527 in 3rd party component Apache Tomcat
    - CVE-2021-23899 in 3rd party component OWASP json-sanitizer
    - CVE-2020-25649 in 3rd party component FasterXML Jackson Databind

  What's new on Oracle Graph Server and Client 21.1.0
  ===================================================

  Main new features on Oracle Graph Server and Client 21.1.0:

    - New graph algorithms focusing on graph-based machine learning: Deep Walk, Supervised GraphWise and PG2VEC
    - Ability to store machine learning models, embeddings (frames) back to the Oracle Database
    - Static permission configuration for the graph server are now stored as roles in the Oracle Database
    - Graph server authentication tokens can now be configured to refresh themselves automatically
    - Property graph views: Ability to query relational tables directly using PGQL queries

  The following problems were fixed in Oracle Graph Server and Client 21.1.0:

    - Bug 32117878 - Delta refresh not working properly if an edge had missing source/destination vertices upon load
    - Bug 32190137 - Delta refresh misbehaving if encountering properties not specified in graph config
    - Bug 31872575 - Importing highlights from older GraphViz versions not working
    - Bug 32004321 - GraphViz: show label on hover has no effect
    - Bug 32321115 - GraphServer class missing in Zeppelin package
    - Bug 32125114 - intermittent IllegalArgumentException when starting server jshell
    - Bug 32148583 - setting token expiration to a long time (e.g 30 days) sets expiration date of token into past
    - Bug 32151756 - GraphViz throws error when deployed into WLS 12.2.1.4.0
    - Bug 32254002 - Connection leak in GraphViz if using PGQL on RDBMS
    - Bug 32260225 - GraphViz does not invalidate HTTP session if PGX token is expired and try to logout

  What's new on Oracle Graph Server and Client 20.4.1
  ===================================================

  This is the first limited patch cycle release on top of version 20.4.0
  which contains fixes for the following issues:

    - Bug 32164434: PGX fails to deploy into Weblogic Server
    - CVE-2020-13943 in 3rd party component Apache Tomcat


  What's new on Oracle Graph Server and Client 20.4.0
  ===================================================

  Main new features on Oracle Graph Server and Client 20.4.0:

    - PGQL's CREATE PROPERTY GRAPH statement now supported by PGX
    - Python client for PGX
    - Client-IP independent load balancing for PGX
    - Best-effort data object tracker for PGX
    - Log into the Graph Visualization application with your Oracle Database credentials

  The following problems were fixed in Oracle Graph Server and Client 20.4.0:

    - Bug 31850135 - GraphViz: Vertex property not correctly marked as selected
    - Bug 31872576 - GraphViz: Self edges are not displayed using Firefox
    - Bug 31351677 - PGX: Unsupported Add/Drop column operation on compressed tables during load
    - Bug 31850139 - PGQL: executePgql(String) should be a query string not an options string
    - Bug 31850144 - PGQL: CREATE PROPERTY GRAPH better error message for multiple labels
    - Bug 31850147 - RDF: SPARQL query with SERVICE expression returns external attributes with null value
    - Bug 31850149 - GraphViz: webapp user pagination settings not being respected
    - Bug 31850152 - GraphViz: graph disappears after re-running a query
    - Bug 31850154 - PGX: Username case-insensitivity not working as expected when logging in
    - Bug 31850157 - SQLcl: PATH macros are not working on SQLcl
    - Bug 31850159 - SQLcl: NullPointerException when turning PGQL mode on without connecting to a database

  What's new on Oracle Graph Server and Client 20.3.0
  ===============================
  Main new features on Oracle Graph Server and Client 20.3.0:

    - PGX authentication, authorization and redaction with Oracle Database as Identity Provider
    - PGX synchronizer: keep data in your partitioned tables in sync with PGX
    - Query results on Graph Visualization webapp can be visualized as Table (Tabular Mode)
    - PGQL plugin for SQLcl
    - Initial release of RDF Server

  The following problems were fixed in Oracle Graph Server and Client 20.3.0:

    - Bug 29396996 - PGX: fixed memory leak during delta refresh
    - Bug 31850057 - PGX: error when using 'id' as property name
    - Bug 31850063 - PGX: Unable to read data - Table definition has changed when reading graph
    - Bug 31850065 - PGX: fixed PG timestamps being read as strings
    - Bug 28408543 - PGQL: Unable to use Oracle Date DataTypes (UnsupportedOperationException)
    - Bug 31202013 - PGX: Setting USE_VERTEX_PROPERTY_VALUE_AS_LABEL in JSON not working
    - Bug 31850074 - GraphViz: error thrown if PGX is using basic scheduler
    - Bug 31850079 - GraphViz: Annotation Mode is not working
    - Bug 31850102 - GraphViz: Right click on a Vertex/Edge is broken in Safari
    - Bug 31698903 - PGX: Preloaded published graphs can be implicitly removed by memory cleanup
    - Bug 31850117 - PGX: Such column list already indexed error during loading from PG
    - Bug 31872635 - Loading from PG format takes too long


  What's new on Oracle Graph Server and Client 20.2.0
  ===============================
  Main new features on Oracle Graph Server and Client 20.2.0:

    - Release of PGQL 1.3
    - Graph Visualization now supports PGQL on RDBMS
    - PL/SQL packages now included in release
    - Initial release of the OCI marketplace image
    - PGQL on RDBMS now supports INSERT/UPDATE/DELETE and any-directed edge patterns
    - Added alias feature to CREATE PROPERTY GRAPH DDL
    - Property Graph names in RDBMS can now be 125 characters long instead of 27 (requires database patch)

  The following problems were fixed in Oracle Graph Server and Client 20.2.0:

    - Bug 31849977 - Fixed PG tables being created with indices with nologging
    - Bug 31175056 - Fixed first delta refresh throws error
    - Bug 31849980 - PGQL DDL: Do not create edge when foreign key is null
    - Bug 31849986 - GraphViz: settings are not displayed correctly in Firefox
    - Bug 31849989 - CREATE PROPERTY GRAPH is not populating V column
    - Bug 31849992 - CREATE PROPERTY GRAPH throws error using Oracle Database 12.2
    - Bug 31849995 - Fixed NoSuchMethodError when deploying into WLS
    - Bug 31849999 - PGX: Do not require DateTime format when loading from DataTime in database
    - Bug 31850004 - Error when loading Oracle Flat File with dates in parallel
    - Bug 31850006 - No .bat files for JShell in client kit
    - Bug 31132819 - PGX: Loading PGX partitioned from 12c throws FeatureNotSupportedException:
                     Operation Custom ID property is not supported


  What's new on Oracle Graph Server and Client 20.1.0
  ===============================
  Main new features on Oracle Graph Server and Client 20.1.0:

    - Initial Release of Graph Visualization webapp
    - Release PGX Partitioned Graph Model - the ability to read directly from relational tables
    - Support for creating custom graph algorithms with Java syntax
    - Support for Autonomous Database
    - Database Backwards Compatilibity
    - Added a CREATE PROPERTY GRAPH statement to PGQL on RDBMS for creating property graphs from existing tables
    - Added support for TOP k CHEAPSET path queries to PGQL on PGX
    - New built-in algorithms
    - Support PGQL FROM keyword in PGQL on RDBMS
    - Disable/enable secondary tables for properties and indices for adjacency vertices on Apache HBase

  The following problems were fixed in Oracle Graph Server and Client 20.1.0:

    - Bug 31849938 - Oracle logo is gone on Graph Visualization webapp
    - Bug 31849942 - Dropdowns to select label text not working on Graph Visualization app
    - Bug 29890674 - CANNOT DO CHECKOUT ON PUBLISHED GRAPHS
    - Bug 29890142 - FILTER ON LOADING: NOT APPLIED WHEN RELOADING A GRAPH WITH A DIFFERENT FILTER
    - Bug 29416268 - REMOVE TINKERPOP 2.X APIS/DEPENDENCIES FROM DAL
    - Bug 30545496 - Graph Visualization 'Settings' display gets messed up
    - Bug 31849990 - has_label() built-in function is not supported
