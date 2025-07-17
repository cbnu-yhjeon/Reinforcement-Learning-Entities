import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random # render_q에서 policy 생성 시 random.choice 사용 가능성

class Renderer:
    def __init__(self, reward_map, goal_state, wall_states, start_state=None):
        self.reward_map = reward_map
        self.goal_state = goal_state
        self.wall_states = wall_states # 리스트로 받음
        self.start_state = start_state
        self.ys = len(self.reward_map)
        self.xs = len(self.reward_map[0])
        self.ax = None

    def set_figure(self, figsize=None):
        if figsize is None:
            figsize = (self.xs * 0.8, self.ys * 0.8)
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)
        ax = self.ax
        ax.clear()
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                       bottom=False, left=False, right=False, top=False)
        ax.set_xticks(np.arange(0, self.xs + 1, 1))
        ax.set_yticks(np.arange(0, self.ys + 1, 1))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.invert_yaxis()
        ax.grid(True, color='lightgray')

    def render_v(self, v_dict=None, policy_dict=None, print_value=True):
        self.set_figure()
        ax = self.ax
        ys, xs = self.ys, self.xs

        if v_dict is not None:
            color_list = ['#FF6347', '#FFFFFF', '#90EE90']
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', color_list)

            v_np = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                if 0 <= state[0] < ys and 0 <= state[1] < xs:
                    v_np[state] = value

            valid_v_values = [val for st, val in v_dict.items() if st not in self.wall_states and st != self.goal_state]
            if not valid_v_values: valid_v_values = [0]
            v_abs_max = max(abs(max(valid_v_values)), abs(min(valid_v_values)), 0.1)
            norm = matplotlib.colors.Normalize(vmin=-v_abs_max, vmax=v_abs_max)

            for r in range(ys):
                for c in range(xs):
                    state = (r, c)
                    if state not in self.wall_states:
                        val_to_color = v_np[state]
                        color = cmap(norm(val_to_color))
                        rect = plt.Rectangle((c, r), 1, 1, color=color, ec='gray')
                        ax.add_patch(rect)

        for r_idx in range(ys):
            for c_idx in range(xs):
                state = (r_idx, c_idx)
                reward_val = self.reward_map[r_idx, c_idx]
                center_x, center_y = c_idx + 0.5, r_idx + 0.5

                if reward_val is not None and reward_val != 0 :
                    txt_color = 'darkgreen' if reward_val > 0 else 'maroon'
                    txt = f'R {reward_val:.1f}'
                    if state == self.goal_state: txt += ' (G)'
                    ax.text(center_x, center_y - 0.2, txt, ha='center', va='center', fontsize=7, color=txt_color, weight='bold')

                if v_dict is not None and state not in self.wall_states and print_value:
                    ax.text(center_x, center_y + (0.2 if reward_val !=0 else 0), f"{v_dict.get(state, 0.0):.2f}",
                            ha='center', va='center', fontsize=8)

                if policy_dict is not None and state not in self.wall_states and state != self.goal_state:
                    if state in policy_dict:
                        actions = policy_dict[state]
                        max_prob = max(actions.values()) if actions else 0
                        max_actions = [act for act, prob in actions.items() if prob == max_prob and prob > 1e-3]
                        arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
                        arrow_offsets = {0: (0, -0.15), 1: (0, 0.15), 2: (-0.15, 0), 3: (0.15, 0)}
                        for action_idx in max_actions:
                            arrow = arrows.get(action_idx, "")
                            offset = arrow_offsets.get(action_idx, (0,0))
                            ax.text(center_x + offset[0], center_y + offset[1] + (0.15 if reward_val !=0 else 0.1), arrow,
                                    ha='center', va='center', fontsize=12, color='black')

                if state in self.wall_states:
                    ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, facecolor='dimgray', hatch='//', ec='black'))

                if self.start_state and state == self.start_state:
                     ax.text(center_x, center_y, "S", ha='center', va='center', fontsize=10, color='blue', weight='bold')
        plt.show()

    def render_q(self, q_dict, print_value=True, show_greedy_policy=True):
        self.set_figure()
        ax = self.ax
        ys, xs = self.ys, self.xs
        action_space = list(range(4))

        if not q_dict: self.render_v(None,None,False); return

        valid_q_values = [val for (st, act), val in q_dict.items() if st not in self.wall_states and st != self.goal_state]
        if not valid_q_values: valid_q_values = [0]
        q_abs_max = max(abs(max(valid_q_values)), abs(min(valid_q_values)), 0.1)
        norm = matplotlib.colors.Normalize(vmin=-q_abs_max, vmax=q_abs_max)
        color_list = ['#FF6347', '#FFFFFF', '#90EE90']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', color_list)

        for r_idx in range(ys):
            for c_idx in range(xs):
                state = (r_idx, c_idx)
                center_x, center_y = c_idx + 0.5, r_idx + 0.5
                reward_val = self.reward_map[r_idx, c_idx]

                if reward_val is not None and reward_val != 0:
                    txt_color = 'darkgreen' if reward_val > 0 else 'maroon'
                    txt = f'R {reward_val:.1f}'
                    if state == self.goal_state: txt += ' (G)'
                    ax.text(center_x, center_y - 0.35, txt, ha='center', va='center', fontsize=6, color=txt_color, weight='bold')

                if self.start_state and state == self.start_state:
                     ax.text(center_x, center_y + (0.35 if reward_val !=0 else 0), "S",
                             ha='center', va='center', fontsize=8, color='blue', weight='bold')

                if state in self.wall_states:
                    ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, facecolor='dimgray', hatch='//', ec='black'))
                    continue

                if state == self.goal_state:
                    fc = 'lightgreen' if (self.reward_map[state] is not None and self.reward_map[state] > 0) else 'lightcoral'
                    ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, facecolor=fc, ec='gray'))
                    continue

                tri_vertices = {
                    0: [(c_idx, r_idx), (c_idx + 1, r_idx), (center_x, center_y)],
                    1: [(c_idx, r_idx + 1), (c_idx + 1, r_idx + 1), (center_x, center_y)],
                    2: [(c_idx, r_idx), (c_idx, r_idx + 1), (center_x, center_y)],
                    3: [(c_idx + 1, r_idx), (c_idx + 1, r_idx + 1), (center_x, center_y)]
                }
                text_offsets = {0: (0,-0.20), 1: (0,0.20), 2: (-0.20,0), 3: (0.20,0)}

                for action_idx in action_space:
                    q_val = q_dict.get((state, action_idx), 0.0)
                    color = cmap(norm(q_val))
                    poly = plt.Polygon(tri_vertices[action_idx], facecolor=color, ec='gray')
                    ax.add_patch(poly)
                    if print_value:
                        offset = text_offsets[action_idx]
                        ax.text(center_x + offset[0], center_y + offset[1], f"{q_val:.2f}",
                                ha='center', va='center', fontsize=7)
        plt.show()

        if show_greedy_policy:
            policy_dict = {}
            for r_idx_p in range(ys):
                for c_idx_p in range(xs):
                    state_p = (r_idx_p, c_idx_p)
                    if state_p in self.wall_states or state_p == self.goal_state: continue
                    action_q_values = [q_dict.get((state_p, act_p), -float('inf')) for act_p in action_space]
                    max_action = np.argmax(action_q_values) if any(val > -float('inf') for val in action_q_values) else random.choice(action_space)
                    probs = {act_p: 0.0 for act_p in action_space}; probs[max_action] = 1.0
                    policy_dict[state_p] = probs
            self.render_v(None, policy_dict)