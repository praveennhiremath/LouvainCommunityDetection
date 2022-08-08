#
# Copyright (C) 2013 - 2022, Oracle and/or its affiliates. All rights reserved.
# ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
#
import argparse
import os
import sys

"""OPGPy shell script."""

def make_red(text):
    """Make the text red."""
    if sys.platform == 'win32':
        return text
    else:
        return "\033[31m{}\033[0m".format(text)


def make_bold(text):
    """Make the text bold."""
    if sys.platform == 'win32':
        return text
    else:
        return "\033[1m{}\033[0m".format(text)


def get_credentials():
    import getpass as prompt
    print()

    # if username is given, only prompt for password
    if args.username is not None:
        password = prompt.getpass('password: ')
        return (args.username, password)

    # no username provided prompt for username and password
    username = input('username: ')
    password = prompt.getpass('password: ')
    return (username, password)


def get_default_kerberos_ticket():
    if sys.platform == 'linux':
        env_path = os.getenv('KRB5CCNAME', None)
        if env_path is not None:
            return (env_path)

        from os import path
        uid = os.getuid()
        kerberos_default_ticket = '/tmp/krb5cc_' + str(uid)
        if path.exists(kerberos_default_ticket):
            return (kerberos_default_ticket)


def get_instance():
    if args.base_url is None:
        return graph_server.get_embedded_instance()

    if args.username is not None:
        (username, password) = get_credentials()
        return graph_server.get_instance(args.base_url, username, password)

    if args.kerberos_ticket is not None:
        return graph_server.get_instance(args.base_url, args.kerberos_ticket)

    # no credentials provided try to authenticate using default Kerberos ticket
    default_kerberos_ticket = get_default_kerberos_ticket()
    if default_kerberos_ticket is not None:
        try:
            print('attempting authentication using default Kerberos ticket detected at ' + default_kerberos_ticket)
            return graph_server.get_instance(args.base_url, default_kerberos_ticket)
        except Exception as e:
            print('could not authenticate using default Kerberos ticket')

    # fallback to prompt for username/password
    (username, password) = get_credentials()
    return graph_server.get_instance(args.base_url, username, password)


parser = argparse.ArgumentParser(prog='opg4py', description='Python PGX client.')

# named arguments
parser.add_argument(
    '--base_url', '-b',
    dest='base_url',
    help='Base URL of the PGX server.'
)
parser.add_argument(
    '--username', '-u',
    dest='username',
    help='Username to use for authentication. Password will be prompted.'
)
parser.add_argument(
    '--kerberos_ticket',
    dest='kerberos_ticket',
    help='Path of kerberos ticket to use for authentication.'
)
parser.add_argument(
    '--idle_timeout',
    dest='idle_timeout',
    type=int,
    help='Number of seconds after which the shell session will time out.',
)
parser.add_argument(
    '--task_timeout',
    dest='task_timeout',
    type=int,
    help='Number of seconds after which tasks submitted by the shell session will time out.',
)
parser.add_argument(
    '--no_banner',
    dest='no_banner',
    action='store_true',
    help='Do not print a banner on shell start.',
)
parser.add_argument(
    '--no_connect',
    dest='no_connect',
    action='store_true',
    help='Do not start/connect to PGX nor create a session.',
)

try:
    args = parser.parse_args()
except SystemExit:
    os._exit(0)

try:
    import pypgx as pgx
    import pypgx.pg.rdbms.graph_server as graph_server
    import opg4py
    from pypgx import setloglevel  # noqa: E402 F401
    from pypgx.api import NAMESPACE_PRIVATE, NAMESPACE_PUBLIC
    from pypgx.api.auth._permission_entity import PgxUser, PgxRole

    if args.no_connect is False:
        try:
            instance = get_instance()
            session = instance.create_session("OPGShell", args.idle_timeout, args.task_timeout, 'seconds')
            analyst = session.create_analyst()
        except Exception as e:
            print('could not connect to server instance:')
            print(e)
            raise SystemExit

except SystemExit:
    os._exit(0)

# Show banner with version information
if args.no_banner is False:
   print(make_red("Oracle Graph Client Shell 22.1.0"))