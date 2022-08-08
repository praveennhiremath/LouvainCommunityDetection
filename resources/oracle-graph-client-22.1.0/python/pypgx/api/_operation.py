#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

import sys
from jnius import autoclass
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import UNHASHABLE_TYPE, ARG_MUST_BE
from typing import Any, List, Optional, Union, TextIO, NoReturn

ByteArrayOutputStream = autoclass('java.io.ByteArrayOutputStream')
PrintStream = autoclass('java.io.PrintStream')


class Operation:
    """An operation is part of an execution plan for executing a PGQL query.

    The execution plan is composed of a tree of operations.
    """

    _java_class = 'oracle.pgx.api.Operation'

    def __init__(self, java_operation) -> None:
        self.java_operation = java_operation
        self.graph_id = java_operation.getGraphId()
        java_operation_type = java_operation.getOperationType()
        self.operation_type = java_handler(java_operation_type.name, [])
        self.cost_estimate = java_operation.getCostEstimate()
        self.total_cost_estimate = java_operation.getTotalCostEstimate()
        self.cardinality_estimate = java_operation.getCardinalityEstimate()
        self.pattern_info = java_operation.getPatternInfo()
        java_children = java_operation.getChildren()
        self.children = [Operation(child) for child in java_children]

    def get_graph_id(self) -> str:
        """Return the graph used in the operation.

        :return: The id of the graph used in the operation
        """
        return java_handler(self.graph_id.toString, [])

    def print(self, file: Optional[TextIO] = None) -> None:
        """Print the current operation and all its children to standard output.

        :param file:File to which results are printed (default is ``sys.stdout``)
        """
        if file is None:
            # We don't have sys.stdout as a default parameter so that any changes
            # to sys.stdout are taken into account by this function
            file = sys.stdout

        # GM-21982: redirect output to the right file
        output_stream = ByteArrayOutputStream()
        print_stream = PrintStream(output_stream, True)
        java_handler(self.java_operation.print, [print_stream])
        print(output_stream.toString(), file=file)
        print_stream.close()
        output_stream.close()

    def get_operation_type(self) -> str:
        """Return the type of operation.

        :return: OperationType of this operation as an enum value
        """
        return self.operation_type

    def get_cost_estimate(self) -> float:
        """
        :return: An estimation of the cost of executing this operation
        """
        return self.cost_estimate

    def get_total_cost_estimate(self) -> float:
        """
        :return: An estimation of the cost of executing this operation and all its children
        """
        return self.total_cost_estimate

    def get_cardinality_estimate(self) -> float:
        """
        :return: An estimation of the cardinality after executing this operation
        """
        return self.cardinality_estimate

    def get_pattern_info(self) -> None:
        """
        :return: An string indicating the pattern that will be matched by this operation
        """
        return self.pattern_info

    def get_children(self) -> List[Union["Operation", Any]]:
        """Return the children of this operation.

        Non leaf operations can have multiple child operations, which will be returned by this
        function.

        :return: A list of operations which are the children of this operation
        """
        return self.children

    def is_same_query_plan(self, other: Union["Operation", str]) -> bool:
        """Check if the query plan with this operation as root node is equal to the query plan
        with 'other' as root node.

        This will only check if the operationType and the pattern are the same for each node in
        both query plans.

        :return: True if both execution plans are the same, false otherwise
        """
        if not isinstance(other, Operation):
            raise TypeError(ARG_MUST_BE.format(arg="other", type=Operation.__name__))
        return java_handler(self.java_operation.isSameQueryPlan, [other.java_operation])

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
