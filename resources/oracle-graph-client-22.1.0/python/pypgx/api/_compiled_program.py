#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from pypgx.api._pgx_graph import PgxGraph
from pypgx.api._pgx_collection import PgxCollection
from pypgx.api._pgx_entity import PgxEntity
from pypgx.api._pgx_map import PgxMap
from pypgx.api._property import PgxProperty
from pypgx.api._scalar import Scalar
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx.api.filters import GraphFilter
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import ARG_MUST_BE, WRONG_NUMBER_OF_ARGS
from pypgx._utils.pgx_types import java_types
from pypgx._utils.pyjnius_helper import PyjniusHelper
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_session import PgxSession

scalar_out_types = (
    'BOOL_OUT',
    'DATE_OUT',
    'DOUBLE_OUT',
    'EDGE_ID_OUT',
    'INT_OUT',
    'LONG_OUT',
    'NODE_ID_OUT',
    'STRING_OUT',
)


class CompiledProgram(PgxContextManager):
    """A compiled Green-Marl program.

    Constructor arguments:

    :param session: Pgx Session
    :type session: PgxSession
    :param java_program: Java compiledProgram
    :type java_program: oracle.pgx.api.CompiledProgram
    """

    _java_class = 'oracle.pgx.api.CompiledProgram'

    def __init__(self, session: "PgxSession", java_program) -> None:
        self._program = java_program
        self.session = session
        self.id = java_program.getId()
        self.compiler_output = java_program.getCompilerOutput()

    def run(self, *argv: Any) -> Dict[str, Optional[int]]:
        """Run the compiled program with the given parameters.

        If the Green-Marl procedure of this compiled program looks like this:
        procedure pagerank(G: Graph, e double, max int, nodePorp){...}

        :param argv: All the arguments required by specified procedure
        :returns: Result of analysis as an AnalysisResult as a dict
        """
        arguments = [self._program]
        arg_types = self._program.getArgumentTypes()
        if len(argv) != len(arg_types):
            raise TypeError(
                WRONG_NUMBER_OF_ARGS.format(expected=len(arg_types), received=len(argv))
            )
        for idx, (arg, arg_type) in enumerate(zip(argv, arg_types)):
            if isinstance(arg, PgxGraph):
                arguments.append(arg._graph)
            elif isinstance(arg, PgxEntity):
                arguments.append(arg._entity)
            elif isinstance(arg, PgxProperty):
                arguments.append(arg._prop)
            elif isinstance(arg, PgxCollection):
                arguments.append(arg._collection)
            elif isinstance(arg, PgxMap):
                arguments.append(arg._map)
            elif isinstance(arg, GraphFilter):
                arguments.append(arg._filter)
            elif arg_type.toString() == 'BOOL_IN':
                arguments.append(java_types["boolean"](arg))
            elif arg_type.toString() == 'INT_IN':
                arguments.append(java_types["integer"](arg))
            elif arg_type.toString() == 'LONG_IN':
                arguments.append(java_types["long"](arg))
            elif arg_type.toString() == 'FLOAT_IN':
                arguments.append(java_types["float"](arg))
            elif arg_type.toString() == 'DOUBLE_IN':
                arguments.append(java_types["double"](arg))
            elif arg_type.toString() in scalar_out_types:
                if not isinstance(arg, Scalar):
                    raise TypeError(
                        ARG_MUST_BE.format(arg='argv[' + str(idx) + ']', type=Scalar.__name__)
                    )
                arguments.append(arg._scalar)
            else:
                arguments.append(arg)
        output = java_handler(PyjniusHelper.runFromPython, arguments)

        analysis_result = {
            'success': output.isSuccess(),
            'canceled': output.isCanceled(),
            'exception': output.getException(),
            'return_value': output.getReturnValue(),
            'execution_time(ms)': output.getExecutionTimeMs(),
        }
        return analysis_result

    def destroy(self) -> None:
        """Free resources on the server taken up by this Program."""
        java_handler(self._program.destroy, [])

    def __repr__(self) -> str:
        return "{}(name: {})".format(self.__class__.__name__, self.id)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._program.equals(other._program)
