import math
import random

MAX_ROUND_NUMBER = 10
AVALABLE_CHOICES = [1,2,3]

class State(object):
    def __int__(self):
        self.current_value = 0.0
        self.current_round_index = 0
        self.cumulative_choices = []

    def get_current_value(self):
        return self.current_value

    def set_current_value(self, value):
        self.current_value = value

    def get_current_round_index(self):
        return self.current_round_index

    def set_current_round_index(self, turn):
        self.current_round_index = turn

    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_cumulative_choices(self, choices):
        self.cumulative_choices = choices

    def is_terminal(self):
        return self.current_round_index == MAX_ROUND_NUMBER

    def compute_reward(self):
        return -abs(1 - self.current_value)

    def get_next_state_with_random_choices(self):
        random_choice = random.choice([choice for choice in AVALABLE_CHOICES])
        next_state = State()
        next_state.set_current_value(self.current_value + random_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])

    def __repr__(self):
        # TODO
        a = 5

class Node(object):
    def __int__(self):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = 0
        self.state = None

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        return len(self.children) == AVALABLE_CHOICES

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)




def tree_policy(node):
    # whether the node is the leaf node
    while node.get_state().is_terminal() == False:
        if node.is_all_extend():
            # selection
            # 选择未被探索的子节点， 如果都探索过则选择UCB值最大的子节点。
            # 没有子节点，就用best_child得到下一个子节点
            node = best_child(node, True)
        else:
            # expand
            sub_node = expand(node)
            return sub_node
    return node

def best_child(node, is_exploration, best_score):
    if is_exploration:
        C = 1 / math.sqrt(2)
    else:
        C = 0
    for sub_node in node.get_children():
        # compute UCB, both exploitation
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.sqrt(2) * math.log(node.get__visit_times()) / sub_node.get_visit_times()
        score = left + C * right

        if score > best_score:
            best_sub_node = sub_node

    return best_sub_node

def expand(node):
    tried_sub_node_states = [sub_node.get_states() for sub_node in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice()
    while new_state in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node

# 在前面expansion出来的节点开始模拟游戏，直到游戏结束，得到该节点的得分。
def default_policy(node):
    current_state = node.get_state()
    while current_state.is_terminal() == False:
        current_state = current_state.get_next_state_with_random_choice()

    final_state_reward = current_state.compute_reward()
    return final_state_reward

def backup(node, reward):
    while node != None:
        node.visit_times_add_one()
        node.quality_value_add_n(reward)
        node = node.parent








