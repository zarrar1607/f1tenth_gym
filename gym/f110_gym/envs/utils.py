from typing import Any, Dict, TypeVar

KeyType = TypeVar("KeyType")


# JAX compatible Enum
# based on https://github.com/epignatelli/jax_enums
@dataclass
class EnumItem:
    name: str
    value: jax.Array
    obj_class: str

    def __str__(self):
        return f"<{self.obj_class}.{self.name}: {self.value}> as PyTreeNode"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))

    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda x: x[idx], self)

    def __eq__(self, other):
        if not isinstance(other, EnumItem):
            raise TypeError(f"Can't compare with non-EnumItem {other}.")
        with jax.ensure_compile_time_eval():
            return jnp.array_equal(self.value, other.value)

    def __ne__(self, other):
        return jnp.logical_not(self.__eq__(other))

    def tree_flatten(self):
        return (self.value), (self.name, self.obj_class)

    @classmethod
    def tree_unflatten(cls, aux, children) -> EnumItem:
        return cls(value=children[0], name=aux[0], obj_class=aux[1])


class JaxEnumMeta(EnumMeta):
    def __new__(mcls, attr_name, bases, attrs, **kwargs):
        for attr_name, value in attrs.items():
            if not attr_name.startswith("_") and not isinstance(value, EnumItem):
                value = EnumItem(
                    value=jnp.asarray(value),
                    name=attr_name,
                    obj_class=attrs["__qualname__"],
                )
                dict.update(attrs, {attr_name: value})
        return super().__new__(mcls, attr_name, bases, attrs)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            not name.startswith("_")
            and not isinstance(value, EnumItem)
            and name == value.name
        ):
            value = EnumItem(
                value=jnp.asarray(value.value.value),
                name=name,
                obj_class=value.value.obj_class,
            )
            return super().__setattr__(name, value)


class JaxEnum(Enum, metaclass=JaxEnumMeta):
    ...


def deep_update(
    mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    """
    Dictionary deep update for nested dictionaries from pydantic:
    https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def distance_transform_1d(vector):
    sorted_indices = np.argsort(np.abs(vector))
    closest_index = sorted_indices[0]
    closest_value = vector[closest_index]

    return closest_value


@jax.jit
def edt(image):
    """
    8 neighborhood exact euclidean distance transform
    """

    @jax.jit
    def replace(v, i, val, inc):
        v = v.at[i].set(val)
        inc += 2
        return v, inc

    @jax.jit
    def dont_replace(v, i, val, inc):
        inc = 1
        return v, inc

    h, w = image.shape
    max_d = h**2 + w**2
    dt = jnp.where(image, max_d, 0)

    # vertical pass
    def dt_1d(vec):
        def pass_fn(vec):
            return vec

        def calc_dist(vec):
            return dist

        dt = jax.lax.cond(
            jnp.count_nonzero(vec) == len(vec),
            pass_fn,
            calc_dist,
            vec,
        )
        return dt

    dt = jax.vmap(
        dt_1d,
        in_axes=[
            1,
        ],
    )(dt)

    # horizontal pass
    for i in range(h):
        # copy row of vertical distances
        dtc = dt[i, :]
        # column positions
        for j in range(w):
            # init min dist
            dist_min = dtc[j]
            # comparee with column position
            for k in range(w):
                # combine vertical and horizontal components
                d = dtc[k] + (k - j) ** 2

                dist_min = jnp.select([dist_min > d, dist_min <= d], [d, dist_min])
                # if dist_min > d:
                #     # new minimum
                #     dist_min = d
            dt.at[i, j].set(dist_min)

    return jnp.sqrt(dt)
