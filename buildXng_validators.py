from collections import Counter
from rich.console import Console
import numpy as np
from typing import Dict, Iterator, List, Set, Tuple


def make_minimum_top_count_validators(
    col_configs: List[List[int]], wire_count=None, color_count=3
) -> List[Tuple[Counter, int, int]]:
    if len(col_configs) < 2:
        if wire_count is None:
            raise ValueError(
                "At least two column configs are needed to build a validator."
            )
        else:
            col = col_configs[0]
            if len(col) * 2 == wire_count:  # normal column
                sc = []
                for cidx, c in enumerate(col):
                    sc.append((Counter({c: 1}), cidx * 2, cidx * 2 + 2))
                return sc
            elif len(col) * 2 + 2 == wire_count:  # small column
                sc = []
                sc.append((Counter({k: 0 for k in range(color_count)}), 0, 1))
                for cidx, c in enumerate(col):
                    sc.append(
                        (
                            Counter({c: 1}),
                            cidx * 2 + 1,
                            cidx * 2 + 3,
                        )
                    )
                sc.append(
                    (
                        Counter({k: 0 for k in range(color_count)}),
                        wire_count - 1,
                        wire_count,
                    )
                )
                return sc
            else:
                raise ValueError(
                    f"Wrong wire count. {wire_count} not matching column length {len(col)}."
                )

    small_col_size = min(len(col_configs[0]), len(col_configs[1]))
    # we determine this because it does change the segment indexing.
    if len(col_configs[0]) > len(col_configs[1]):
        is_small_col = False
        node_count = len(col_configs[0])
        wire_count = node_count * 2

        max_depth = wire_count - 1
    else:
        is_small_col = True
        node_count = len(col_configs[0])
        wire_count = (node_count + 1) * 2

        max_depth = wire_count - 1

    if len(col_configs) < max_depth + 1:
        max_depth = len(col_configs) - 1
    min_segment_counters = []
    # we start at the max_depth layer
    max_depth_config = col_configs[max_depth]
    for node_idx, dot_color in enumerate(col_configs[max_depth]):
        is_small_col = len(max_depth_config) == small_col_size

        # current_counter for this node
        counter = Counter({k: 0 if k != dot_color else 1 for k in range(color_count)})

        if is_small_col:
            start_idx = node_idx * 2 + 1
            end_idx = start_idx + 2
        else:
            start_idx = node_idx * 2
            end_idx = start_idx + 2
        # now iteratively jump to the previous layer
        for depth in range(max_depth - 1, -1, -1):
            col_config_at_depth = col_configs[depth]
            is_small_col = len(col_config_at_depth) == small_col_size

            if is_small_col:
                color_peek_start = max(0, start_idx // 2 - 1)
                color_peek_end = (end_idx) // 2
                count_for_depth = Counter(
                    col_config_at_depth[color_peek_start:color_peek_end]
                )
                for dot_color, v in count_for_depth.items():
                    if counter[dot_color] < v:
                        counter[dot_color] = v
                # update start_idx and end_idx for next depth
                start_idx = start_idx - 1
                end_idx = end_idx + 1
            else:
                color_peek_start = max(0, (start_idx - 1) // 2)
                color_peek_end = (end_idx // 2) + 1
                count_for_depth = Counter(
                    col_config_at_depth[color_peek_start:color_peek_end]
                )
                for dot_color, v in count_for_depth.items():
                    if counter[dot_color] < v:
                        counter[dot_color] = v
                # update start_idx and end_idx for next depth
                start_idx = start_idx - 1
                end_idx = end_idx + 1
            start_idx = max(0, start_idx)
            end_idx = min(wire_count, end_idx)
        min_segment_counters.append((counter, start_idx, end_idx))
    return min_segment_counters


def counter_to_vec(c: Counter[int], color_count):
    vec = np.zeros(color_count, dtype=np.int8)
    for col in range(color_count):
        vec[col] = c.get(col, 0)
    return vec


def counter_to_rich(c: Counter[int], color_count):
    # color int to color string
    def int_to_color_str(i: int) -> str:
        # Simple mapping for demonstration; adjust as needed
        return {
            0: "red",
            1: "blue",
            2: "green",
            3: "yellow",
            4: "magenta",
            5: "cyan",
            7: "white",
        }[int(i)]

    if isinstance(c, Counter):
        atoms = []
        for col in range(color_count):
            v = c.get(col, 0)
            color = int_to_color_str(col)
            atoms.append(f"[{color}]{v}[/{color}]")
        return "".join(atoms)
    elif isinstance(c, np.ndarray):
        atoms = []
        for col in range(color_count):
            v = c[col]
            color = int_to_color_str(col)
            atoms.append(f"[{color}]{v}[/{color}]")
        return "".join(atoms)
    else:
        raise ValueError("Unsupported counter type for rich conversion.")


def build_min_max_validator(
    bconfig: List[List[int]],
    fima: np.ndarray,
    color_count=3,
) -> Tuple[
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Counter,
]:
    wire_count = int(np.sum(fima))
    is_small_col = len(bconfig[0]) * 2 + 2 == wire_count

    upto = min(len(bconfig), wire_count * 2)
    validators = [
        make_minimum_top_count_validators(
            bconfig[:i], wire_count, color_count=color_count
        )
        for i in range(1, upto + 1)
    ]
    # index the segments in a dictionnary
    segment_index = {}

    for vd in validators:
        for counter, start_idx, end_idx in vd:
            seg = (start_idx, end_idx)
            if seg in segment_index:
                segment_index[seg] = counter.__or__(segment_index[seg])
            else:
                segment_index[seg] = counter

    if is_small_col:
        segment_index[(0, 1)] = Counter({k: 0 for k in range(color_count)})
        segment_index[(wire_count - 1, wire_count)] = Counter(
            {k: 0 for k in range(color_count)}
        )
    # propagate constraints to larger segments (the triangle matrix tricks)
    for _ in range(wire_count**2):
        for k, min_contraint in list(segment_index.items()):
            start_idx, end_idx = k
            if start_idx == 0 and end_idx != wire_count:
                for k2, c2 in list(segment_index.items()):
                    if end_idx == k2[0]:

                        k2_end_idx = k2[1]

                        if (
                            0,
                            k2_end_idx,
                        ) in segment_index:  # can be missing because no depth
                            add = (min_contraint + c2) | segment_index[(0, k2_end_idx)]
                            segment_index[(0, k2_end_idx)] = add
                        else:
                            add = min_contraint + c2
                            segment_index[(0, k2_end_idx)] = add

    # convert all counters to numpy arrays for easier manipulation
    segment_index = {
        k: counter_to_vec(c1, color_count=color_count)
        for k, c1 in segment_index.items()
    }
    # minimum assortmen viable
    if (0, wire_count) in segment_index:
        mina = segment_index[(0, wire_count)]
    else:
        mina = fima

    # compute max constraints by segments by applying fima - segment count
    # and segment complement
    # we do first all the edge segments
    max_constraints = {}
    for (start_idx, end_idx), min_contraint in segment_index.items():
        if start_idx == 0:
            max_count = fima - min_contraint
            # find the max constraint interval by removing k from the (0, wire_count) segment (the global one)
            max_interval = (end_idx, wire_count)
            max_constraints[max_interval] = max_count
        elif end_idx == wire_count and start_idx != 0:
            max_count = fima - min_contraint
            max_interval = (0, start_idx)
            max_constraints[max_interval] = max_count
    # finally process the full range segment (0, wire_count)
    max_constraints[(0, wire_count)] = fima

    # if we have a small col , we need to calculate the min for the (0,1 ) and (wire_count-1, wire_count)
    # those two can be deduced from the full segment max - min of the complement
    if is_small_col:
        # 0--1 segment
        left_min = segment_index.get((1, wire_count), np.zeros(color_count, dtype=int))
        max_constraints[(0, 1)] = fima - left_min

        # wire_count-1 -- wire_count segment (the final one)
        right_min = segment_index.get(
            (0, wire_count - 1), np.zeros(color_count, dtype=int)
        )
        max_constraints[(wire_count - 1, wire_count)] = fima - right_min

    # now calculate some max constraint for the non-edge segments
    for (start_idx, end_idx), min_contraint in segment_index.items():
        if start_idx != 0 and end_idx != wire_count:
            # like (4,6 )
            # seek for the left border element (0, 6) max
            left_max = max_constraints.get((0, end_idx), None)
            # seek for the left border element (0,4 ) min
            left_min = segment_index.get((0, start_idx), None)
            if left_max is not None and left_min is not None:
                max_constraints[(start_idx, end_idx)] = left_max - left_min
            else:
                pass
    all_missing_colors = Counter({k: 0 for k in range(color_count)})

    for k, min_contraint in segment_index.items():
        if k in max_constraints:
            if np.any(max_constraints[k] - min_contraint < 0):
                # we calculate the negative diff
                diff = max_constraints[k] - min_contraint
                # get the indices where negative
                negative_indices = np.where(diff < 0)[0]
                missing_colors = Counter(
                    {
                        int(color_idx): -int(diff[color_idx])
                        for color_idx in negative_indices
                    }
                )
                all_missing_colors = all_missing_colors.__or__(missing_colors)
    return segment_index, max_constraints, all_missing_colors


def build_min_max_validator_2(
    bconfig: List[List[int]],
    fima: np.ndarray,
    color_count=3,
) -> Tuple[
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Counter,
]:
    wire_count = int(np.sum(fima))
    is_small_col = len(bconfig[0]) * 2 + 2 == wire_count

    upto = min(len(bconfig), wire_count * 2)
    validators = [
        make_minimum_top_count_validators(bconfig[:i], wire_count, color_count)
        for i in range(1, upto + 1)
    ]

    def calc_upper_bound(mc_v, n, fima):
        max_c_v = np.zeros(color_count, dtype=int)

        mc_vsum = mc_v.sum()
        for color_idx in range(color_count):
            max_c_v[color_idx] = n - (mc_vsum - mc_v[color_idx])
        max_c_v = np.min([fima, max_c_v], axis=0)
        return max_c_v

    def print_count_with_max(mcounts):
        for mc, s, e in mcounts:
            mc_v = counter_to_vec(mc, color_count=color_count)
            n = e - s
            max_c_v = np.zeros(color_count, dtype=int)

            mc_vsum = mc_v.sum()
            for color_idx in range(color_count):
                max_c_v[color_idx] = n - (mc_vsum - mc_v[color_idx])
            max_c_v = np.min([fima, max_c_v], axis=0)
            console.print(
                f"Segment {s}-{e} : {counter_to_rich(mc)}<  <{counter_to_rich(max_c_v)}"
            )

    constraints_set: Dict[Tuple[int, int], np.ndarray] = {}
    # we will add constraints in a specific order to propagate better
    for mcs in validators:
        for min_constraint, s, e in mcs:
            seg = (s, e)
            mc_v = counter_to_vec(min_constraint, color_count=color_count)
            insert_min_counter(constraints_set, seg, mc_v, color_count)

    # now we order the constraints set by segment_start , then segment_end
    # console.print("final lower bound calculation:")
    # current_s = -1
    # buff = ""
    # for (s, e), min_constraint in sorted(
    #     constraints_set.items(), key=lambda x: (x[0][0], x[0][1])
    # ):
    #     if current_s != s:
    #         console.print(buff)
    #         current_s = s
    #         buff = ""
    #     n = e - s
    #     buff += f"({s:02d}-{e:02d}) {counter_to_rich(min_constraint)}< "
    # console.print(buff)

    # we build the upper bound set :
    upped_bound_set: Dict[Tuple[int, int], np.ndarray] = {}
    for (s, e), min_constraint in sorted(
        constraints_set.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        n = e - s
        upper_bound = calc_upper_bound(min_constraint, n, fima)
        add_upper_bound(
            upped_bound_set, (s, e), upper_bound, constraints_set, color_count
        )

    for seg, upper_bound in list(upped_bound_set.items()):
        add_upper_bound(
            upped_bound_set,
            seg,
            upper_bound,
            constraints_set,
            color_count,
            force_propagate=True,
        )
    # console.print("Final upper bound calculation:")
    # current_s = -1
    # buff = ""
    # for (s, e), min_constraint in sorted(
    #     constraints_set.items(), key=lambda x: (x[0][0], x[0][1])
    # ):
    #     if current_s != s:
    #         console.print(buff)
    #         current_s = s
    #         buff = ""
    #     upper_bound = upped_bound_set[(s, e)]
    #     buff += f"({s:02d}-{e:02d}) {counter_to_rich(min_constraint)}<{counter_to_rich(upper_bound)} "
    # console.print(buff)

    all_missing_colors = Counter({k: 0 for k in range(color_count)})

    # minimum assortment viable
    # check if any upper bound has negative components

    for (s, e), upper_bound in upped_bound_set.items():
        min_constraint = constraints_set[(s, e)]
        if np.any(upper_bound - min_constraint < 0):
            # console.print(
            #     f"[red]Inconsistent constraints for segment {s}-{e} : {counter_to_rich(min_constraint)}<{counter_to_rich(upper_bound)}[/red]"
            # )
            # we calculate the negative diff
            diff = upper_bound - min_constraint
            # get the indices where negative
            negative_indices = np.where(diff < 0)[0]
            missing_colors = Counter(
                {
                    int(color_idx): -int(diff[color_idx])
                    for color_idx in negative_indices
                }
            )
            all_missing_colors = all_missing_colors.__or__(missing_colors)
    return constraints_set, upped_bound_set, all_missing_colors


def build_col_iterator(
    colors_required: List[int],
    top_min_validators: Dict[Tuple[int, int], np.ndarray],
    top_max_validators: Dict[Tuple[int, int], np.ndarray],
    bottom_min_validators: Dict[Tuple[int, int], np.ndarray],
    bottom_max_validators: Dict[Tuple[int, int], np.ndarray],
    wire_count: int,
    color_count=3,
    allowed_tops: Dict[Tuple[int, int], Set[bytes]] = None,
) -> Iterator[Tuple[List[int], List[int], List[int]]]:
    is_small_col = len(colors_required) * 2 + 2 == wire_count

    def node_pos_to_wire_pos(node_pos):
        if is_small_col:
            return (node_pos * 2 + 1, node_pos * 2 + 3)
        else:
            return (node_pos * 2, node_pos * 2 + 2)

    def build_(at_node_pox, Tseq, Bseq, Iseq, tseq_count):
        # we build color 2 by 2 thread for each node.
        # we will get first the index of the wire for the top and bottom
        _, end_wire_idx = node_pos_to_wire_pos(at_node_pox)
        # then we grab the min& max validators for this segment
        min_val = top_min_validators.get((0, end_wire_idx), None)
        max_val = top_max_validators.get((0, end_wire_idx), None)

        if bottom_min_validators is not None:
            min_val_b = bottom_min_validators.get((0, end_wire_idx), None)
        else:
            min_val_b = None
        if bottom_max_validators is not None:
            # #TODO : check this
            max_val_b = bottom_max_validators.get((0, end_wire_idx), None)
            # because of the interleaved nature ,
            # this might be always return None.
            # TODO : verify this logic
        else:
            max_val_b = None

        for LT in range(color_count):
            for RT in range(color_count):
                if (
                    LT != colors_required[at_node_pox]
                    and RT != colors_required[at_node_pox]
                ):
                    continue
                chunk_count = np.zeros(color_count, dtype=int)
                chunk_count[LT] += 1
                chunk_count[RT] += 1

                current_count = tseq_count + chunk_count
                # check top validators
                if min_val is not None:
                    if np.any(current_count - min_val < 0):
                        continue
                if max_val is not None:
                    if np.any(max_val - current_count < 0):
                        continue

                # check allowed tops
                if allowed_tops is not None:
                    candidate_top = bytes(Tseq + [LT, RT])
                    if candidate_top not in allowed_tops.get((0, end_wire_idx), set()):
                        continue

                # check bottom validators
                if min_val_b is not None:
                    if np.any(current_count - min_val_b < 0):
                        continue
                if max_val_b is not None:
                    if np.any(max_val_b - current_count < 0):
                        continue

                # now its a flip game
                for flip in set([LT != RT, False]):

                    tseq_b = Tseq + [LT, RT]
                    if flip:
                        bseq_b = Bseq + [RT, LT]
                    else:
                        bseq_b = Bseq + [LT, RT]

                    if at_node_pox + 1 < len(colors_required):
                        # not leaf yet
                        yield from build_(
                            at_node_pox + 1, tseq_b, bseq_b, Iseq, current_count
                        )
                    else:
                        # leaf
                        yield tseq_b, bseq_b, Iseq, current_count

    if is_small_col:
        for color_start_thread in range(color_count):
            # TODO : if c match a parent bound init
            if (0, 1) in top_max_validators:
                max_val = top_max_validators[(0, 1)]
                if max_val[color_start_thread] <= 0:
                    # can't pick this color
                    continue
            if bottom_max_validators is not None and ((0, 1) in bottom_max_validators):
                max_val = bottom_max_validators[(0, 1)]
                if max_val[color_start_thread] <= 0:
                    # can't pick this color
                    continue
            tseq = [color_start_thread]
            bseq = [color_start_thread]
            iseq = []
            tseq_count = np.zeros(color_count, dtype=int)
            tseq_count[color_start_thread] += 1
            for tseq_b, bseq_b, iseq_b, tseq_count_b in build_(
                0, tseq, bseq, iseq, tseq_count
            ):
                for color_end in range(color_count):

                    tseq_count = np.zeros(color_count, dtype=int)
                    tseq_count[color_end] += 1

                    with_end_count = tseq_count_b + tseq_count
                    # check top min validator
                    if (0, wire_count) in top_min_validators:
                        min_val = top_min_validators[(0, wire_count)]
                        if np.any(with_end_count - min_val < 0):
                            continue
                    # check top max validator
                    if (0, wire_count) in top_max_validators:
                        max_val = top_max_validators[(0, wire_count)]
                        if np.any(max_val - with_end_count < 0):
                            continue

                    # check allowed tops
                    if allowed_tops is not None:
                        candidate_top = bytes(tseq_b + [color_end])
                        if candidate_top not in allowed_tops.get(
                            (0, wire_count), set()
                        ):
                            continue
                    # check bottom min validator
                    if (
                        bottom_min_validators is not None
                        and (0, wire_count) in bottom_min_validators
                    ):
                        min_val = bottom_min_validators[(0, wire_count)]
                        if np.any(with_end_count - min_val < 0):
                            continue
                    # check bottom max validator
                    if (
                        bottom_max_validators is not None
                        and (0, wire_count) in bottom_max_validators
                    ):
                        max_val = bottom_max_validators[(0, wire_count)]
                        if np.any(max_val - with_end_count < 0):
                            continue
                    tseq_f = tseq_b + [color_end]
                    bseq_f = bseq_b + [color_end]
                    yield tseq_f, bseq_f, iseq_b
    else:
        for t, b, i, _ in build_(0, [], [], [], np.zeros(color_count, dtype=int)):
            yield t, b, i


def insert_min_counter(constraints_set, seg, mc_v, color_count):
    """
    Insert a minimum counter (lower bound) into the constraints_set dictionary,
    propagating the changes to other segments as necessary.
    """

    s, e = seg

    changed_something = False
    # first we insert or update the segment
    if seg in constraints_set:
        new_value = np.max([constraints_set[seg], mc_v], axis=0)
        old_value = constraints_set[seg]
        if not np.array_equal(new_value, old_value):
            constraints_set[seg] = np.max([constraints_set[seg], mc_v], axis=0)
            changed_something = True
    else:
        constraints_set[seg] = mc_v
        changed_something = True

    if not changed_something:
        return
    else:
        # we need to propagate the change to other segments
        # with calculation of the "sum max" for some others
        for (other_s, other_e), other_mc in list(constraints_set.items()):

            if other_s == e:
                # we have a (s,e) and (e, other_e) so we can update (s, other_e)
                combined_seg = (s, other_e)
                combined_min = constraints_set[seg] + other_mc
                insert_min_counter(
                    constraints_set, combined_seg, combined_min, color_count
                )
            elif other_e == s:
                # we have a (other_s, s) and (s, e) so we can update (other_s, e)
                combined_seg = (other_s, e)
                combined_min = other_mc + constraints_set[seg]
                insert_min_counter(
                    constraints_set, combined_seg, combined_min, color_count
                )
            elif other_e == e and other_s < s:
                # like we are 4-6 and other is 2-6 (combined), we update 2-6 with  2-4 (the pad) + 4-6
                combined_seg = (other_s, e)
                pad_seg = (other_s, s)
                pad_mc = constraints_set.get(pad_seg, np.zeros(color_count, dtype=int))
                combined_min = pad_mc + constraints_set[seg]
                insert_min_counter(
                    constraints_set, combined_seg, combined_min, color_count
                )
            elif other_s == s and other_e > e:
                # like we are 4-6 and other is 4-8 (combined), we update 4-8 with 4-6 + 6-8 (the pad)
                combined_seg = (s, other_e)
                pad_seg = (e, other_e)
                pad_mc = constraints_set.get(pad_seg, np.zeros(color_count, dtype=int))
                combined_min = constraints_set[seg] + pad_mc
                insert_min_counter(
                    constraints_set, combined_seg, combined_min, color_count
                )


def add_upper_bound(
    upper_bound_set,
    seg,
    upper_bound,
    lower_bounds_set,
    color_count,
    force_propagate=False,
):
    s, e = seg
    change_something = False
    if (s, e) in upper_bound_set:
        previous_value = upper_bound_set[(s, e)]
        new_value = np.min([upper_bound_set[(s, e)], upper_bound], axis=0)
        if not np.array_equal(previous_value, new_value):
            change_something = True
            upper_bound_set[(s, e)] = new_value

    else:
        upper_bound_set[(s, e)] = upper_bound
        change_something = True
    if not change_something and not force_propagate:
        return
    # we need to propagate the change to other segments
    # and now we can calculate joins to propagate upper bounds
    for (other_s, other_e), other_ub in list(upper_bound_set.items()):

        if other_e == s:
            # we are (s, e) and we have a (other_s, s) (on my left side) so we can update (other_s, s) upper bound
            # like we are (4-6) and other is (2-4) , we can update (2-4) with (2-4) <= ((2-6)max - (4-6)min)
            combined_seg = (other_s, e)  # (2,6)

            if combined_seg in upper_bound_set:
                combined_ub = upper_bound_set[combined_seg]
                min_constraint = lower_bounds_set.get(
                    (s, e), np.zeros(color_count, dtype=int)
                )

                add_upper_bound(
                    upper_bound_set,
                    (other_s, s),
                    np.min([other_ub, combined_ub - min_constraint], axis=0),
                    lower_bounds_set,
                    color_count,
                )

            else:
                # should not happen because we build from smaller to larger
                pass
        elif other_s == e:
            # we are (s, e) and we have a (e, other_e) (on my right side) so we can update (other_s, other_e) upper bound
            # like we are (4-6) and other is (6-8) , we can update (6-8) with (6-8) <= ((4-8)max - (4-6)min)
            combined_seg = (s, other_e)  # (4,8)

            if combined_seg in upper_bound_set:
                combined_ub = upper_bound_set[combined_seg]
                min_constraint = lower_bounds_set.get(
                    (s, e), np.zeros(color_count, dtype=int)
                )

                add_upper_bound(
                    upper_bound_set,
                    (e, other_e),
                    np.min([other_ub, combined_ub - min_constraint], axis=0),
                    lower_bounds_set,
                    color_count,
                )

            else:
                # should not happen because we build from smaller to larger
                pass
        elif other_e == e and other_s < s:
            # we are (s, e) and other is (other_s, e) ,
            #     (s               e)
            # (other_s       other_e)
            #
            # we can update (other_s, s) with :  (other_s, s) <= (other_s, other_e)max - (s,e)min

            combined_seg = (other_s, e)  # (2,6)
            pad_seg = (other_s, s)  # (2,4)

            if combined_seg in upper_bound_set:
                combined_ub = upper_bound_set[combined_seg]
                min_constraint = lower_bounds_set.get(
                    (s, e), np.zeros(color_count, dtype=int)
                )

                add_upper_bound(
                    upper_bound_set,
                    pad_seg,
                    np.min(
                        [
                            upper_bound_set.get(pad_seg, combined_ub),
                            combined_ub - min_constraint,
                        ],
                        axis=0,
                    ),
                    lower_bounds_set,
                    color_count,
                )

        elif other_s == s and other_e > e:
            # we are (s, e) and other is (s, other_e) ,
            #  (s        e)
            #  (other_s       other_e)
            #
            # we can update (e, other_e) with (e, other_e) <= (s, other_e)max - (s,e)min

            combined_seg = (s, other_e)  # (4,8)
            pad_seg = (e, other_e)  # (6,8)

            if combined_seg in upper_bound_set:
                combined_ub = upper_bound_set[combined_seg]
                min_constraint = lower_bounds_set.get(
                    (s, e), np.zeros(color_count, dtype=int)
                )

                add_upper_bound(
                    upper_bound_set,
                    pad_seg,
                    np.min(
                        [
                            upper_bound_set.get(pad_seg, combined_ub),
                            combined_ub - min_constraint,
                        ],
                        axis=0,
                    ),
                    lower_bounds_set,
                    color_count,
                )


if __name__ == "__main__":
    console = Console(force_interactive=False)
    bconfig = [
        # "CCBABABCC",
        "CCAAAACC",
        "CCAABAACC",
        "CCAAAACC",
        "BCAAAAACB",
        "CCAAAACC",
    ]
    fima = np.array([11, 3, 4], dtype=np.int8)
    bconfig = [
        "BCAA",
        "BAA",
        "ACAA",
        "BAA",
        "ACAB",
        "BAA",
        "ACAA",
        "BAA",
        "ACAA",
        "BAA",
        "ACAA",
        "BAA",
    ]
    fima = np.array([5, 2, 1], dtype=np.int8)
    # B C A A
    #  B A A
    # A C A A
    #  B A A
    # A C A B
    color_count = 3
    wire_count = max(len(b) for b in bconfig) * 2
    # current user assortment is always fixed (Final Maximal Assortment)
    console.print("fima:", counter_to_rich(fima))

    column_to_work = 0

    # convert the config into integers.
    bconfig_ints = [[ord(c) - ord("A") for c in bc] for bc in bconfig]

    lower_bound_set, upper_bound_set, missing_colors = build_min_max_validator_2(
        bconfig_ints[column_to_work:], fima, color_count=color_count
    )

    console.print("missing :", counter_to_rich(missing_colors))
    console.print("Final upper bound calculation:")
    current_s = -1
    buff = ""
    for (s, e), min_constraint in sorted(
        lower_bound_set.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        if current_s != s:
            console.print(buff)
            current_s = s
            buff = ""
        upper_bound = upper_bound_set[(s, e)]
        buff += f"({s:02d}-{e:02d}) {counter_to_rich(min_constraint)}<{counter_to_rich(upper_bound)} "
    console.print(buff)

    # now iterate solutions
    raw_solutions_iter = build_col_iterator(
        bconfig_ints[column_to_work],
        lower_bound_set,
        upper_bound_set,
        None,
        None,
        color_count=color_count,
        wire_count=wire_count,
    )

    raw_solutions = list(raw_solutions_iter)
    print(len(raw_solutions))
    #### quit , rest is discarded for now
    quit()
    min_constraints, max_constraints, missing_colors = build_min_max_validator(
        bconfig_ints[column_to_work:],
        fima,
        color_count=color_count,
    )

    if missing_colors.total() > 0 or True:
        console.print(
            f"[red]Warning: infeasible configuration, missing colors:[/red] {(missing_colors)}"
        )

        # now print all constraints
        for k, min_contraint in min_constraints.items():
            if k in max_constraints:
                max_constraint = counter_to_rich(max_constraints[k])
                if np.any(max_constraints[k] - min_contraint < 0):
                    # we calculate the negative diff
                    diff = max_constraints[k] - min_contraint
                    # get the indices where negative
                    negative_indices = np.where(diff < 0)[0]
                    missing_colors = Counter(
                        {
                            int(color_idx): -int(diff[color_idx])
                            for color_idx in negative_indices
                        }
                    )

                    max_constraint += f"  [red](infeasible){missing_colors}[/red]"
            else:
                max_constraint = "N/A"
            console.print(f"{k} > {counter_to_rich(min_contraint)}   !{max_constraint}")

        for k, c2 in max_constraints.items():
            if k in min_constraints:
                pass
            else:
                console.print(f"{k}         !{counter_to_rich(c2)}")
        quit()

    raw_solutions_iter = build_col_iterator(
        bconfig_ints[column_to_work],
        min_constraints,
        max_constraints,
        color_count=color_count,
        wire_count=wire_count,
    )

    raw_solutions = list(raw_solutions_iter)
    print(len(raw_solutions))
