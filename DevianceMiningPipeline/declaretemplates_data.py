"""
These are used for data-aware declare deviance mining.
Each template gives back: locations activations, which were fulfilled and activations, which were violated.
No short-circuiting of conditions, which could previously be done in other templates
"""


def template_not_responded_existence(trace, event_set):
    """
    if A occurs and Cond holds, B can never occur
    :param trace:
    :param event_set:
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    # All activations are either fulfilled or all are violated!

    if event_1 in trace:
        if event_2 not in trace:
            return len(trace[event_1]), False
        else:
            # Return all indices as violations!
            # violation at every activation!
            return -1, False
    else:
        # Vacuous if no event_1
        return 0, True


def template_not_responded_existence_data(trace, event_set):
    """
    if A occurs and Cond holds, B can never occur
    :param trace:
    :param event_set:
    :return:

    - number of fulfillments if all fulfilled, second if vacuous, third fulfilled activations, fourth violated activations
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    # All activations are either fulfilled or all are violated!

    if event_1 in trace:
        if event_2 not in trace:
            return len(trace[event_1]), False, trace[event_1], [] # Third one is the place of activations!
        else:
            # Return all indices as violations!
            # violation at every activation!
            return -1, False, [], trace[event_2]
    else:
        # Vacuous if no event_1
        return 0, True, [], []


def template_not_precedence(trace, event_set):
    """
    if B occurs and Cond holds, A cannot have occurred before
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]
    if event_2 in trace:
        if event_1 not in trace:
            return len(trace[event_2]), False # frequency for every activation
        else:
            # For every B occurred, have to check that A hasn't occurred before.
            # This means, that last B must have occurred before first A
            first_pos_event_1 = trace[event_1][0] # first A position
            last_pos_event_2 = trace[event_2][-1] # last B position
            if last_pos_event_2 < first_pos_event_1:
                # todo: check frequency condition
                count = len(trace[event_2])
                return count, False
            return -1, False
    else:
        # Vacuous if no event_2
        return 0, True



def template_not_precedence_data(trace, event_set):
    """
    if B occurs and Cond holds, A cannot have occurred before
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    fulfillments = []
    violations = []

    # Find first A, all B's which are before A are fulfilled, others violated

    event_1 = event_set[0]
    event_2 = event_set[1]
    if event_2 in trace:
        if event_1 not in trace:
            return len(trace[event_2]), False, trace[event_2], []  # Then all are fulfillments
        else:
            first_pos_event_1 = trace[event_1][0] # first A position
            # Check every pos_2
            for pos_event_2 in trace[event_2]:
                if pos_event_2 < first_pos_event_1:
                    # pos_event_1 is AFTER B, therefore precedence does not hold
                    fulfillments.append(pos_event_2)
                else:
                    # pos_event_1 is BEFORE B, therefore precedence holds!
                    violations.append(pos_event_2)

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations
    else:
        # Vacuous if no event_2
        return 0, True, [], []


def template_not_chain_response(trace, event_set):
    """
    if A occurs and Cond holds, B cannot be executed next
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            # for each event_1_position, check if there is event_2_position after
            for pos1 in event_1_positions:
                if pos1+1 in event_2_positions:
                    # this means that there is!
                    return -1, False

            # None of pos1+1 was in event2, therefore there is no chain_response
            return len(event_1_positions), False

        else:
            return len(trace[event_1]), False  # no response for event1, therefore all fulfilled

    return 0, True  # todo, vacuity


def template_not_chain_response_data(trace, event_set):
    """
    if A occurs and Cond holds, B cannot be executed next

    For every A, check if B comes next. If not, then violted
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    fulfillments = []
    violations = []

    if event_1 in trace:
        if event_2 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            # for each event_1_position, check if there is event_2_position after
            for pos1 in event_1_positions:
                if pos1+1 in event_2_positions:
                    # this means that there is!
                    violations.append(pos1)
                else:
                    fulfillments.append(pos1)

            if len(violations) > 0:
                # at least one violation,
                return -1, False, fulfillments, violations
            else:
            # None of pos1+1 was in event2, therefore there is no chain_response
                return len(fulfillments), False, fulfillments, violations

        else:
            return len(trace[event_1]), False, trace[event_1], []  # no response for event1, therefore all fulfilled

    return 0, True, [], []  # todo, vacuity


def template_not_chain_precedence(trace, event_set):
    """
    if B occurs and Cond holds, A cannot have occurred immediately before
    :param trace:
    :param event_set:
    :return:
    """

    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_2 in trace:
        if event_1 in trace:
            # Every event2 must NOT be chain preceded by event1
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            # for each event_2_position, check if there is event_1_position before
            for pos2 in event_2_positions:
                if pos2-1 in event_1_positions:
                    # this means that there is!
                    return -1, False

            count = len(event_2_positions)
            return count, False
        else:
            return len(trace[event_2]), False  # no response for event1

    return 0, True  # todo, vacuity


def template_not_chain_precedence_data(trace, event_set):
    """
    if B occurs and Cond holds, A cannot have occurred immediately before
    :param trace:
    :param event_set:
    :return:
    """
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    fulfillments = []
    violations = []

    if event_2 in trace:
        if event_1 in trace:
            # Every event2 must NOT be chain preceded by event1
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            # for each event_2_position, check if there is event_1_position before
            for pos2 in event_2_positions:
                if pos2-1 in event_1_positions:
                    # this means that there is!
                    violations.append(pos2)
                else:
                    fulfillments.append(pos2)


            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            return len(trace[event_2]), False, trace[event_2], []  # no response for event1

    return 0, True, [], []  # todo, vacuity


def template_alternate_precedence(trace, event_set):
    """
      precedence(A, B) template indicates that event B
      should occur only if event A has occurred before.

      Alternate condition:
      "events must alternate without repetitions of these events in between"

      :param trace:
      :param event_set:
      :return:
      """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]
    if event_2 in trace:
        if event_1 in trace:
            # Go through two lists, one by one
            # first events pos must be before 2nd lists first pos etc...
            # A -> A -> B -> A -> B

            # efficiency check
            event_1_count = len(trace[event_1])
            event_2_count = len(trace[event_2])

            # There has to be more or same amount of event A's compared to B's
            if event_2_count > event_1_count:
                return -1, False

            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]

            # Go through all event 2's, check that there is respective event 1.
            # Find largest event 1 position, which is smaller than event 2 position

            # implementation
            # Check 1-forward, the 1-forward has to be greater than event 2 and current one has to be smaller than event2

            event_1_ind = 0
            for i, pos2 in enumerate(event_2_positions):
                # find first in event_2_positions, it has to be before next in event_1_positions

                while True:
                    if event_1_ind >= len(event_1_positions):
                        # out of preceding events, but there are still event 2's remaining.
                        return -1, False

                    next_event_1_pos = None

                    if event_1_ind < len(event_1_positions) - 1:
                        next_event_1_pos = event_1_positions[event_1_ind + 1]

                    event_1_pos = event_1_positions[event_1_ind]

                    if next_event_1_pos:
                        if event_1_pos < pos2 and next_event_1_pos > pos2:
                            # found the largest preceding event
                            event_1_ind += 1
                            break
                        elif event_1_pos > pos2 and next_event_1_pos > pos2:
                            # no event larger
                            return -1, False
                        else:
                            event_1_ind += 1


                    else:
                        # if no next event, check if current is smaller
                        if event_1_pos < pos2:
                            event_1_ind += 1
                            break
                        else:
                            return -1, False  # since there is no smaller remaining event

            count = len(event_2_positions)
            return count, False


        else:
            # impossible because there has to be at least one event1 with event2
            return -1, False

    return 0, True  # todo: vacuity condition!!


def template_alternate_precedence_data(trace, event_set):
    """
      precedence(A, B) template indicates that event B
      should occur only if event A has occurred before.

      Alternate condition:
      "events must alternate without repetitions of these events in between"

      :param trace:
      :param event_set:
      :return:
      """

    # exactly 2 event
    assert (len(event_set) == 2)

    fulfillments = []
    violations = []

    event_1 = event_set[0]
    event_2 = event_set[1]
    if event_2 in trace:
        if event_1 in trace:
            # Go through two lists, one by one
            # first events pos must be before 2nd lists first pos etc...
            # A -> A -> B -> A -> B

            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]
            # FOR every EVENT 2: Going before in the log: THERE MUST BE EVENT 1 BEFORE EVENT 2!
            # Keep track of largest event_1_pos before current event_2_pos. Sorting array.
            merged = []
            event_1_ind = 0
            event_2_ind = 0
            while event_1_ind < len(event_1_positions) and event_2_ind < len(event_2_positions):
                if event_1_positions[event_1_ind] < event_2_positions[event_2_ind]:
                    merged.append((1, event_1_positions[event_1_ind]))
                    event_1_ind += 1
                else:
                    merged.append((2, event_2_positions[event_2_ind]))
                    event_2_ind += 1

            # Merge leftovers
            while event_1_ind < len(event_1_positions):
                merged.append((1, event_1_positions[event_1_ind]))
                event_1_ind += 1

            while event_2_ind < len(event_2_positions):
                merged.append((2, event_2_positions[event_2_ind]))
                event_2_ind += 1

            # Go through array, at every point check if (2, x). If 2, then check if previous is 2 or 1.
            for i in range(len(merged)):
                if merged[i][0] == 2:
                    if i == 0:
                        # If first one, then previous cant be 1. violation.
                        violations.append(merged[i][1])
                    elif merged[i-1][0] == 1:
                        # If is not same, then no violation
                        fulfillments.append(merged[i][1])
                    else:
                        # Therefore if previous is same... then violation
                        violations.append(merged[i][1])

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            # impossible because there has to be at least one event1 with event2. Therefore all activations are violated
            return -1, False, [], trace[event_2]

    return 0, True, [], []  # todo: vacuity condition!!


def template_alternate_response(trace, event_set):
    """
    If there is A, it has to be eventually followed by B.
    Alternate: there cant be any further A until first next B
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:

            event_2_ind = 0

            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]

            for i, pos1 in enumerate(event_1_positions):
                # find first in event_2_positions, it has to be before next in event_1_positions
                next_event_1_pos = None
                if i < len(event_1_positions) - 1:
                    next_event_1_pos = event_1_positions[i + 1]

                while True:
                    if event_2_ind >= len(event_2_positions):
                        # out of response events
                        return -1, False

                    if event_2_positions[event_2_ind] > pos1:
                        # found first greater than event 1 pos
                        # check if it is smaller than next event 1
                        if next_event_1_pos and event_2_positions[event_2_ind] > next_event_1_pos:
                            # next event 2 is after next event 1..
                            return -1, False
                        else:
                            # consume event 2 and break out to next event 1
                            event_2_ind += 1
                            break

                    event_2_ind += 1

            count = len(event_1_positions)
            return count, False
            # every event 2 position has to be after respective event 1 position and before next event 2 position



        else:
            return -1, False

    # Vacuously
    return 0, True


def template_alternate_response_data(trace, event_set):
    """
    If there is A, it has to be eventually followed by B.
    Alternate: there cant be any further A until first next B
    :param trace:
    :param event_set:
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    fulfillments = []
    violations = []

    event_1 = event_set[0]
    event_2 = event_set[1]
    if event_1 in trace:
        if event_2 in trace:
            # Go through two lists, one by one
            # first events pos must be before 2nd lists first pos etc...
            # A -> A -> B -> A -> B

            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]
            # FOR every EVENT 2: Going before in the log: THERE MUST BE EVENT 1 BEFORE EVENT 2!
            # Keep track of largest event_1_pos before current event_2_pos. Sorting array.
            merged = []
            event_1_ind = 0
            event_2_ind = 0
            while event_1_ind < len(event_1_positions) and event_2_ind < len(event_2_positions):
                if event_1_positions[event_1_ind] < event_2_positions[event_2_ind]:
                    merged.append((1, event_1_positions[event_1_ind]))
                    event_1_ind += 1
                else:
                    merged.append((2, event_2_positions[event_2_ind]))
                    event_2_ind += 1

            # Merge leftovers
            while event_1_ind < len(event_1_positions):
                merged.append((1, event_1_positions[event_1_ind]))
                event_1_ind += 1

            while event_2_ind < len(event_2_positions):
                merged.append((2, event_2_positions[event_2_ind]))
                event_2_ind += 1


            # Go through array, at every point check if (2, x). If 2, then check if next is 2 or 1.
            for i in range(len(merged)):
                if merged[i][0] == 1:
                    if i == len(merged) - 1:
                        # Last in list! Will not be responded! Violated.
                        violations.append(merged[i][1])
                    elif merged[i+1][0] == 2:
                        # If no violation is not same, then no violation
                        fulfillments.append(merged[i][1])
                    else:
                        # Therefore if previous is same... then violation
                        violations.append(merged[i][1])

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            # impossible because there has to be at least one event2 with event1. Therefore all activations are violated
            return -1, False, [], trace[event_1]

    return 0, True, [], []  # todo: vacuity condition!!


def template_chain_precedence(trace, event_set):  # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_2 in trace:
        if event_1 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            if len(event_1_positions) < len(event_2_positions):
                return -1, False  # impossible to fulfill

            # For every pos2, check if pos2-1 is in event1 set
            for pos2 in event_2_positions:
                if pos2-1 not in event_1_positions:
                    # Then no possible to preceed!!
                    return -1, False

            count = len(event_2_positions)
            return count, False
        else:
            return -1, False  # no response for event1

    return 0, True  # todo, vacuity


def template_chain_precedence_data(trace, event_set):  # exactly 2 event
    """
    if B occurs and Cond holds, A must have occurred immedi-ately before
    :param trace:
    :param event_set:
    :return:
    """
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    violations = []
    fulfillments = []

    if event_2 in trace:
        if event_1 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])
            # For every pos2, check if pos2-1 is in event1 set
            for pos2 in event_2_positions:
                if pos2 - 1 not in event_1_positions:
                    # Then no possible to preceed!!
                    violations.append(pos2)
                else:
                    fulfillments.append(pos2)

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            return -1, False, [], trace[event_2]  # no response for event1

    return 0, True, [], []  # todo, vacuity


def template_chain_response(trace, event_set):
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            if len(event_1_positions) > len(event_2_positions):
                return -1, False  # impossible to fulfill

            for pos1 in event_1_positions:
                if pos1 + 1 not in event_2_positions:
                    # Exists event 1 which is not chained by event 2
                    return -1, False

            count = len(event_1_positions)
            return count, False

        else:
            return -1, False  # no response for event1

    return 0, True  # todo, vacuity


def template_chain_response_data(trace, event_set):
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    violations = []
    fulfillments = []

    if event_1 in trace:
        if event_2 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = set(trace[event_1])
            event_2_positions = set(trace[event_2])

            for pos1 in event_1_positions:
                if pos1 + 1 not in event_2_positions:
                    # Exists event 1 which is not chained by event 2
                    violations.append(pos1)
                else:
                    fulfillments.append(pos1)


            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            return -1, False, [], trace[event_1]  # no response for event1

    return 0, True, [], []  # todo, vacuity


def template_precedence(trace, event_set):
    """
    precedence(A, B) template indicates that event B
    should occur only if event A has occurred before.
    :param trace:
    :param event_set:
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    violations = []
    fulfillments = []

    if event_2 in trace:
        if event_1 in trace:
            first_pos_event_1 = trace[event_1][0]
            first_pos_event_2 = trace[event_2][0]
            # All event B's, which are before first event A are violated. Every other is fulfilled.
            if first_pos_event_1 < first_pos_event_2:
                # todo: check frequency condition
                count = len(trace[event_2])
                return count, False
            else:
                # first position of event 2 is before first event 1
                return -1, False

        else:
            # impossible because there has to be at least one event1 with event2
            return -1, False

    # Vacuously fulfilled
    return 0, True


def template_precedence_data(trace, event_set):
    """
    precedence(A, B) template indicates that event B
    should occur only if event A has occurred before.
    :param trace:
    :param event_set:
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    violations = []
    fulfillments = []

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_2 in trace:
        if event_1 in trace:
            first_pos_event_1 = trace[event_1][0]
            event_2_positions = trace[event_2]
            # All event B's, which are before first event A are violated. Every other is fulfilled.

            for event_2_pos in event_2_positions:
                if event_2_pos < first_pos_event_1:
                    # There is event
                    violations.append(event_2_pos)
                else:
                    fulfillments.append(event_2_pos)

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            # impossible because there has to be at least one event1 with event2
            return -1, False, [], trace[event_2]

    # Vacuously fulfilled
    return 0, True, [], []



def template_not_response(trace, event_set):
    """
    if A occurs and Cond holds, B can never occur
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            first_pos_event_1 = trace[event_1][0]
            last_pos_event_2 = trace[event_2][-1]
            # Last event 2, must be before first event 1
            if last_pos_event_2 < first_pos_event_1:
                # todo: check frequency counting How to count fulfillments? min of A and B?
                count = len(trace[event_1])
                return count, False
            else:
                # last event2 is before event1
                return -1, False
        else:
            # impossible for event 2 to be after event 1 if there is no event 2
            return len(trace[event_1]), False

    return 0, True  # not vacuity atm..




def template_not_response_data(trace, event_set):
    """
    if A occurs and Cond holds, B can never occur
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    violations = []
    fulfillments = []

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            event_1_positions = trace[event_1]
            last_pos_event_2 = trace[event_2][-1]

            # Go through all event 1, if it is after last_pos_event_2 then fulfilled, otherwise false

            for pos1 in event_1_positions:
                if pos1 < last_pos_event_2:
                    violations.append(pos1)
                else:
                    fulfillments.append(pos1)

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations

        else:
            # impossible for event 2 to be after event 1 if there is no event 2
            return len(trace[event_1]), False, trace[event_1], []

    return 0, True, [], [] # not vacuity atm..


def template_response(trace, event_set):
    """
    If event B is the response of event A, then when event
    A occurs, event B should eventually occur after A.
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            last_pos_event_1 = trace[event_1][-1]
            last_pos_event_2 = trace[event_2][-1]
            if last_pos_event_2 > last_pos_event_1:
                # todo: check frequency counting How to count fulfillments? min of A and B?
                count = len(trace[event_1])
                return count, False

            else:
                # last event2 is before event1
                return -1, False
        else:
            # impossible for event 2 to be after event 1 if there is no event 2
            return -1, False

    return 0, True  # not vacuity atm..


def template_response_data(trace, event_set):
    """
    If event B is the response of event A, then when event
    A occurs, event B should eventually occur after A.
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    violations = []
    fulfillments = []

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            event_1_positions = trace[event_1]
            last_pos_event_2 = trace[event_2][-1]

            for pos_1 in event_1_positions:
                if pos_1 < last_pos_event_2:
                    fulfillments.append(pos_1)
                else:
                    violations.append(pos_1)

            if len(violations) > 0:
                return -1, False, fulfillments, violations
            else:
                return len(fulfillments), False, fulfillments, violations


        else:
            # impossible for event 2 to be after event 1 if there is no event 2
            return -1, False, [], trace[event_1]

    return 0, True, [], []  # not vacuity atm..


def template_responded_existence(trace, event_set):
    """
    The responded existence(A, B) template specifies that
    if event A occurs, event B should also occur (either
        before or after event A).
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            return len(trace[event_1]), False
        else:
            return -1, False

    return 0, True  # 0, if vacuity condition


def template_responded_existence_data(trace, event_set):
    """
    The responded existence(A, B) template specifies that
    if event A occurs, event B should also occur (either
        before or after event A).
    :return:

    All either true or false!
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            return len(trace[event_1]), False, trace[event_1], []
        else:
            return -1, False, [], trace[event_1]

    return 0, True, [], []  # 0, if vacuity condition



# Does order matter in template?
template_order = {
    "alternate_precedence": True,
    "alternate_response": True,
    "chain_precedence": True,
    "chain_response": True,
    "responded_existence": True,
    "response": True,
    "precedence": True,
    "not_responded_existence": True,
    "not_precedence": True,
    "not_response": True,
    "not_chain_response": True,
    "not_chain_precedence": True
}


not_templates = ["not_responded_existence",
                 "not_precedence",
                 "not_response",
                 "not_chain_response",
                 "not_chain_precedence"]


templates = ["alternate_precedence", "alternate_response", "chain_precedence", "chain_response",
             "responded_existence", "response", "precedence"]


template_sizes = {"alternate_precedence": 2,
                  "alternate_response": 2,
                  "chain_precedence": 2,
                  "chain_response": 2,
                  "responded_existence": 2,
                  "response": 2,
                  "precedence": 2,
                  "not_responded_existence": 2,
                  "not_precedence": 2,
                  "not_response": 2,
                  "not_chain_response": 2,
                  "not_chain_precedence": 2
                  }


def apply_data_template(template_str, trace, event_set):
    template_map = {
        "alternate_precedence": template_alternate_precedence_data,
        "alternate_response": template_alternate_response_data,
        "chain_precedence": template_chain_precedence_data,
        "chain_response": template_chain_response_data,
        "responded_existence": template_responded_existence_data,
        "response": template_response_data,
        "precedence": template_precedence_data,
        "not_responded_existence": template_not_responded_existence_data,
        "not_precedence": template_not_precedence_data,
        "not_response": template_not_response_data,
        "not_chain_response": template_not_chain_response_data,
        "not_chain_precedence": template_not_chain_precedence_data
    }

    lower = template_str.lower()

    if lower in template_map:
        return template_map[lower](trace["events"], event_set)
    else:
        raise Exception("Template not found")


def apply_template(template_str, trace, event_set):
    template_map = {
        "alternate_precedence": template_alternate_precedence,
        "alternate_response": template_alternate_response,
        "chain_precedence": template_chain_precedence,
        "chain_response": template_chain_response,
        "responded_existence": template_responded_existence,
        "response": template_response,
        "precedence": template_precedence,
        "not_responded_existence": template_not_responded_existence,
        "not_precedence": template_not_precedence,
        "not_response": template_not_response,
        "not_chain_response": template_not_chain_response,
        "not_chain_precedence": template_not_chain_precedence
    }

    lower = template_str.lower()

    if lower in template_map:
        return template_map[lower](trace["events"], event_set)
    else:
        raise Exception("Template not found")
