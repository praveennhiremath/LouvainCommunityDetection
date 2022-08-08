@ECHO OFF
IF "%1" NEQ "--base_url" (
  IF "%1" NEQ "-b" (
    IF "%1" NEQ "--no_connect" (
      echo --base_url or -b is a required argument when connecting to the PGX server using the remote shell, alternatively use --no_connect to skip PGX server connection & (exit /b 1)
    )
  )
)
SET DIR=%~dp0
SET CP=%DIR%..\lib\opg-sombrero_jshell-22.1.0.jar;%DIR%..\lib\*;%DIR%..\conf;
SET PGX_TMP_DIR=%DIR%..\pgx\tmp_data

java -cp "%CP%;%OPG_CLASSPATH%" oracle.pg.jshell.SombreroShell %*
