"""
Testcases for variables:
- Input
- Output
- ActiveLearning
- VariableGroup
"""

import numpy as np
from profit.util.variable import Variable


def test_input():
    from profit.util.variable import InputVariable

    NAME = "v"
    KIND = "uniform"
    CONSTRAINTS = (2, 5)
    SIZE = (3, 1)
    DTYPE = np.float64
    VARIABLE_DICT = {
        "name": NAME,
        "kind": KIND,
        "constraints": CONSTRAINTS,
        "size": SIZE,
        "value": None,
        "dtype": DTYPE,
    }

    v1 = Variable.create(NAME, KIND, SIZE, constraints=CONSTRAINTS)
    constraints_str = ", ".join(str(c) for c in CONSTRAINTS)
    v2 = Variable.create_from_str(
        NAME,
        SIZE,
        "{kind}({constraints})".format(kind=KIND, constraints=constraints_str),
    )
    v3 = InputVariable(NAME, KIND, SIZE, CONSTRAINTS)
    variables = (v1, v2, v3)
    assert all(v.name == NAME for v in variables)
    assert all(v.size == SIZE for v in variables)
    assert all(v.dtype == DTYPE for v in variables)
    for v in variables:
        v.generate_values()
    VARIABLE_DICT["value"] = variables[0].value
    assert all(
        CONSTRAINTS[0] <= v.value.min() and v.value.max() <= CONSTRAINTS[1]
        for v in variables
    )
    assert np.all(
        v.named_value == np.array(v.value, dtype=[(v.name, v.dtype)]) for v in variables
    )
    assert np.all(v.as_dict() == VARIABLE_DICT for v in variables)
    assert np.all(
        v[key] == value for key, value in VARIABLE_DICT.items() for v in variables
    )

    CONST = 1.4
    v4 = Variable.create_from_str(NAME, SIZE, f"{CONST}")
    assert v4.kind.lower() == "constant"
    v4.generate_values()
    assert np.all(v4.value == np.ones(SIZE) * CONST)


def test_output():
    from profit.util.variable import OutputVariable

    NAME = "f"
    KIND = "Output"
    DEPENDENT = "t"
    SIZE = (3, 1)
    DTYPE = np.float64
    VARIABLE_DICT = {
        "name": NAME,
        "kind": KIND,
        "dependent": DEPENDENT,
        "size": SIZE,
        "value": None,
        "dtype": DTYPE,
    }

    DEPENDENT_SIZE = (200, 1)
    DEPENDENT_VARIABLE = Variable.create_from_str(
        "t", DEPENDENT_SIZE, "Independent(0, 100, 200)"
    )
    v1 = Variable.create(NAME, KIND, SIZE, dependent=DEPENDENT_VARIABLE)
    v2 = Variable.create_from_str(NAME, SIZE, f"{KIND}({DEPENDENT})")
    v3 = OutputVariable(NAME, KIND, SIZE, DEPENDENT)
    variables = (v1, v2, v3)
    for v in variables:
        v.resolve_dependent(DEPENDENT_VARIABLE)
    assert all(v.name == NAME for v in variables)
    assert all(v.size == (SIZE[0], DEPENDENT_SIZE[0]) for v in variables)
    assert all(v.dtype == DTYPE for v in variables)
    VARIABLE_DICT["value"] = variables[0].value
    assert all(
        v.dependent[0].as_dict() == DEPENDENT_VARIABLE.as_dict() for v in variables
    )
    assert np.all(
        v.named_value == np.array(v.value, dtype=[(v.name, v.dtype)]) for v in variables
    )
    assert np.all(v.as_dict() == VARIABLE_DICT for v in variables)
    assert np.all(
        v[key] == value for key, value in VARIABLE_DICT.items() for v in variables
    )


def test_active_learning():
    from profit.util.variable import ActiveLearningVariable

    NAME = "v"
    KIND = "ActiveLearning"
    DISTR = "uniform"
    CONSTRAINTS = (0, 2)
    SIZE = (10, 1)
    DTYPE = np.float64
    VARIABLE_DICT = {
        "name": NAME,
        "kind": KIND,
        "distr": DISTR,
        "constraints": CONSTRAINTS,
        "size": SIZE,
        "value": None,
        "dtype": DTYPE,
    }

    v1 = Variable.create(NAME, KIND, SIZE, distr=DISTR, constraints=CONSTRAINTS)
    constraints_str = ", ".join(str(c) for c in CONSTRAINTS)
    v2 = Variable.create_from_str(
        NAME,
        SIZE,
        "{kind}({constraints})".format(kind=KIND, constraints=constraints_str),
    )
    v3 = ActiveLearningVariable(NAME, KIND, SIZE, DISTR, CONSTRAINTS)
    variables = (v1, v2, v3)

    assert all(v.name == NAME for v in variables)
    assert all(v.size == SIZE for v in variables)
    assert all(v.dtype == DTYPE for v in variables)

    VARIABLE_DICT["value"] = variables[0].value
    assert all(v.distr == DISTR for v in variables)
    assert np.all(
        v.named_value == np.array(v.value, dtype=[(v.name, v.dtype)]) for v in variables
    )
    assert np.all(v.as_dict() == VARIABLE_DICT for v in variables)
    assert np.all(
        v[key] == value for key, value in VARIABLE_DICT.items() for v in variables
    )

    # Distribute prediction points for AL according to distribution
    Xpred = v1.create_Xpred(SIZE)
    assert np.all(Xpred == np.linspace(*CONSTRAINTS, SIZE[0]).reshape(SIZE))

    DISTR = "log"
    CONSTRAINTS = (1e-2, 1e3)
    v4 = ActiveLearningVariable(NAME, KIND, SIZE, DISTR, CONSTRAINTS)
    Xpred = v4.create_Xpred(SIZE)
    assert np.all(
        Xpred
        == CONSTRAINTS[0]
        * np.exp(
            (np.log(CONSTRAINTS[1]) - np.log(CONSTRAINTS[0]))
            * np.linspace(0, 1, SIZE[0])
        ).reshape(SIZE)
    )


def test_variablegroup():
    from profit.util.variable import VariableGroup

    SIZE = (3, 1)
    var = {
        "v": "Uniform(0, 1)",
        "u": "ActiveLearning(3, 4)",
        "t": "Independent(1, 5, 100)",
        "f": "Output(t)",
    }

    variables = [
        Variable.create_from_str(name, SIZE, v_str) for name, v_str in var.items()
    ]
    group = VariableGroup(samples=SIZE[0])
    group.add(variables)

    assert all(group[v.name] is v for v in variables)
    # assert group.all  # Not working yet
    variables[1].value = np.random.random(SIZE)
    variables[-1].value = np.random.random(variables[-1].size)

    assert all(
        np.all(group.input[:, i] == v.value.flat) for i, v in enumerate(variables[:2])
    )
    assert all(np.all(group.named_input[v.name] == v.value) for v in variables[:2])
    assert np.all(group.output == variables[-1].value)
    assert np.all(group.named_output["f"] == variables[-1].value)
    assert group.formatted_output.dtype.names == tuple(
        f"f({t})" for t in variables[2].value.flat
    )
    assert all(group.as_dict[v.name] == v.as_dict() for v in variables)
    group.delete_variable(0)
    assert all(
        np.all(group.input[:, i] == v.value.flat) for i, v in enumerate(variables[1:2])
    )
    group.delete_sample(0)
    assert (
        group.input.shape[0]
        == group.output.shape[0]
        == variables[1].value.shape[0]
        == variables[-1].value.shape[0]
        == SIZE[0] - 1
    )
