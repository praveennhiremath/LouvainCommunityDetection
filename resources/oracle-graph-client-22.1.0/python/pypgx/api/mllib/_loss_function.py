#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#


from pypgx._utils.item_converter import convert_to_java_type
from pypgx._utils.pgx_types import SUPERVISED_LOSS_FUNCTIONS
from typing import Any, List


class LossFunction(object):
    """Abstract LossFunction class that represent loss functions"""

    _java_class = 'oracle.pgx.config.mllib.loss.LossFunction'

    def __init__(self, java_arg_list: List[Any]) -> None:
        self._java_arg_list = java_arg_list


class SoftmaxCrossEntropyLoss(LossFunction):
    """Softmax Cross Entropy loss for multi-class classification"""

    _java_class = 'oracle.pgx.config.mllib.loss.SoftmaxCrossEntropyLoss'

    def __init__(self) -> None:
        super().__init__([])

    def __repr__(self) -> str:
        return "%s" % (self.__class__.__name__)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)


class SigmoidCrossEntropyLoss(LossFunction):
    """Sigmoid Cross Entropy loss for binary classification"""

    _java_class = 'oracle.pgx.config.mllib.loss.SigmoidCrossEntropyLoss'

    def __init__(self):
        super().__init__([])

    def __repr__(self) -> str:
        return "%s" % (self.__class__.__name__)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)


class DevNetLoss(LossFunction):
    """Deviation loss for anomaly detection"""

    _java_class = 'oracle.pgx.config.mllib.loss.DevNetLoss'

    def __init__(self, confidence_margin: float, anomaly_property_value: bool) -> None:
        """
        :param confidence_margin: confidence margin
        :param anomaly_property_value: property value that represents the anomaly
        """

        anomaly_property_value = convert_to_java_type(anomaly_property_value)
        super().__init__([confidence_margin, anomaly_property_value])

    def __repr__(self) -> str:
        return "%s" % (self.__class__.__name__)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._java_arg_list[0] == other._java_arg_list[0] and self._java_arg_list[1].equals(
            other._java_arg_list[1]
        )


def _get_loss_function(loss_fn_name: str) -> LossFunction:
    """Retrieve LossFunction object that can be instantiated no constructor argument"""

    loss_fn_name = loss_fn_name.upper()
    if loss_fn_name not in SUPERVISED_LOSS_FUNCTIONS.keys():
        raise ValueError(
            'Loss function string (%s) must be of the following types: %s'
            % (loss_fn_name, ', '.join(SUPERVISED_LOSS_FUNCTIONS.keys()))
        )

    loss_fn = None
    for subclass in LossFunction.__subclasses__():
        if subclass.__name__ is SUPERVISED_LOSS_FUNCTIONS[loss_fn_name]:
            loss_fn = subclass()

    return loss_fn
