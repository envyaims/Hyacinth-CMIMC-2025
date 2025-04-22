'''
NOTE: A LOT OF EXPLORATION LOGIC HAS BEEN REMOVED AS THEY ARE APPLICABLE TO 
ANOTHER COMPETITION. MORE SPECIFICALLY, THE FOLLOWING HAVE BEEN REMOVED:
  - the vast majority of scout logic, calculation of exploration zones
  - the vast majority of escape logic, everything but bugnav trigger for escape
  - waypoint-optimized nav for medium difficulty

However, the remaining bot is still on par with the original, averaging slightly
less than the uncensored version (avg 21.73 -> avg 16.33). If you want info on
the removed logic, there is a general outline of the strategy in #flood which
you can read for inspiration.
'''

from bots.bot import Bot
from typing import List, Tuple, Set, Dict
import random
import numpy as np
import math
from collections import deque


view_radius = 8
grid_size = 512

# tolerance for barrier detection
BARRIER_EPS = 1
BARRIER_DELTA = 30.0
NOTIFIER_CUTOFF = 150  # distance threshold to go back to original peak


def sign(x: int) -> int:
    return -1 if x < 0 else (0 if x == 0 else 1)


# idk
class SubmissionBot(Bot):
    """
    Bot that explores along a diagonal, broadcasts its best-known peak,
    listens to others to share the global peak, then may converge on it.
    """
    def __init__(self, index: int, difficulty: int):
        global NOTIFIER_CUTOFF
        self.index = index
        random.seed(self.index)

        self.EASY_CUTOFF = 705
        self.MEDIUM_CUTOFF = 870
        self.HARD_CUTOFF = 950
        self.grid_size = 512
        self.difficulty = difficulty
        if (self.difficulty == 0):
            NOTIFIER_CUTOFF = 200
        self.diags = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
        self.diagsSquare = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        self.dir_x, self.dir_y = self.diags[index % 2]
        self.local_history = deque(maxlen=40)

        # bugnav
        self.safe_turns = 5              # ← how many turns ahead we consider “impassable”
        self.impassable = set()           # set of (x,y) globally known to be too low
        self.bugging = False
        self.bug_dir = (0, 0)

        # State
        self.round = 0
        self.best_height = None
        self.best_abs_x = 0
        self.best_abs_y = 0
        self.turns_since_improve = 0
        self.x = 0
        self.y = 0
        self.last_randomize_round = -10
        self.scout = False
        self.scout_count = 10

        # barrier memory (global y-coords of cracks)
        self.barriers: Set[int] = set()
        self.barrier_heights: Dict[int, float] = {}
        self.openings: Dict[int, Set[int]] = {}

        # global average height estimator
        self.sum_heights = 0.0
        self.count_heights = 0
        self.avg_height = None

        self.last_randomize_round = -10

        # misc explore
        self.startedZigZagX = False
        self.startedZigZagY = False
        self.zigzag = 10
        self.zigzagState = self.index % 2 == 0
        self.cease_explore = False
        self.bug_failures = 0
        self.last_bug_failure_round = -50
        self.escape_mode = 0

        # spread / notifier class bots
        self.original_peak: Tuple[int,int] = None
        self.notifier_phase = False
        self.notified = False
        self.offset_chosen = False
        self.offset = (0, 0)


    def cheb_dist(self, x1, y1, x2, y2):
        dx = (x2 - x1) % grid_size
        if dx > grid_size//2: dx -= grid_size
        dy = (y2 - y1) % grid_size
        if dy > grid_size//2: dy -= grid_size
        return max(abs(dx), abs(dy))

    def step(
        self,
        height: np.ndarray,
        neighbors: List[Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        if (self.difficulty == 0):
            return self.stepEasy(height, neighbors)
        elif (self.difficulty == 1):
            return self.stepMedium(height, neighbors)
        elif (self.difficulty == 2):
            return self.stepHard(height, neighbors)


    # MARK: EASY MODE LOGIC
    def stepEasy(
        self,
        height: np.ndarray,
        neighbors: List[Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        

        VIEWPOINT_AVERAGE_HEIGHT = 0

        # ————————————————
        # 0) MARK “IMPASSABLE” TILES
        # flood height at end of this step will be ~ self.round+1,
        # so anything with terrain ≤ current_flood + safe_turns is unsafe.
        threshold = float(self.round + self.safe_turns)
        H, W = height.shape
        for i in range(H):
            for j in range(W):
                h = float(height[i, j])
                VIEWPOINT_AVERAGE_HEIGHT += h
                if h <= threshold:
                    # compute the global coord of (i,j)
                    dx = i - view_radius
                    dy = j - view_radius
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    self.impassable.add((gx, gy))
        
        VIEWPOINT_AVERAGE_HEIGHT /= H * W
        old_best = self.best_height


        center_h = float(height[view_radius, view_radius])
        self.sum_heights += VIEWPOINT_AVERAGE_HEIGHT
        self.count_heights += 1
        self.avg_height = self.sum_heights / self.count_heights

        # ————————————————
        # 1) local scan
        center_h = float(height[view_radius, view_radius])
        if self.best_height is None:
            self.best_height = center_h
            self.best_abs_x = self.x
            self.best_abs_y = self.y
            self.turns_since_improve = 0

        ind = np.unravel_index(np.argmax(height, axis=None), height.shape)
        local_h = float(height[ind])
        dx_local = ind[0] - view_radius
        dy_local = ind[1] - view_radius

        rolling_max = max(self.local_history) if self.local_history else -float('inf')
        just_found_slope = False

        if local_h > rolling_max:
            just_found_slope = True
            if local_h > self.best_height:
                self.best_height = local_h
                self.best_abs_x = (self.x + dx_local) % grid_size
                self.best_abs_y = (self.y + dy_local) % grid_size
                self.turns_since_improve = 0
        else:
            self.turns_since_improve += 1

        self.local_history.append(local_h)


        # neighbor messages
        saw_another_bot = False
        for nx, ny, m in neighbors:
            if (nx == -self.dir_x and ny == -self.dir_y):
                continue

            h_code = (m >> 18) & 0x3FFF
            dx_code = (m >> 9) & 0x1FF
            dy_code = m & 0x1FF
            rel_dx = dx_code - 256
            rel_dy = dy_code - 256
            neigh_abs_x = (self.x + nx) % 512
            neigh_abs_y = (self.y + ny) % 512
            nbx = (neigh_abs_x + rel_dx) % 512
            nby = (neigh_abs_y + rel_dy) % 512
            if h_code > self.best_height:
                self.cease_explore = False
                self.best_height = float(h_code)
                self.best_abs_x = nbx
                self.best_abs_y = nby
                self.turns_since_improve = 0

            saw_another_bot = True
        

        if 320 < self.round < 400 and old_best is not None and self.best_height > old_best \
            and self.original_peak and not self.notified:
            ox, oy = self.original_peak
            d1 = self.cheb_dist(self.x, self.y, ox, oy)
            d2 = self.cheb_dist(ox, oy, self.best_abs_x, self.best_abs_y)
            d_direct = self.cheb_dist(self.x, self.y, self.best_abs_x, self.best_abs_y)
            extra = d1 + d2 - d_direct
            if extra <= NOTIFIER_CUTOFF:
                self.notifier_phase = True

        raw_dx = (self.best_abs_x - self.x) % self.grid_size
        raw_dy = (self.best_abs_y - self.y) % self.grid_size
        dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
        dxg = raw_dx - grid_size if raw_dx > grid_size//2 else raw_dx
        dyg = raw_dy - grid_size if raw_dy > grid_size//2 else raw_dy

        # ————————————————
        # 2) ORIGINAL MOVEMENT DECISION
        self.round += 1
        flood_h = float(self.round)
        if self.notifier_phase and self.original_peak:
            tx, ty = self.original_peak
            dxn = (tx - self.x + grid_size) % grid_size
            dyn = (ty - self.y + grid_size) % grid_size
            sx = sign(dxn - (grid_size if dxn>grid_size//2 else 0))
            sy = sign(dyn - (grid_size if dyn>grid_size//2 else 0))
            movex, movey = sx, sy
            # arrived?
            if self.x == tx and self.y == ty:
                self.notifier_phase = False
                self.notified = True
        elif self.round <= 0:
            movex, movey = self.dir_x, self.dir_y
            if just_found_slope:
                movex, movey = sign(dx_local), sign(dy_local)
        elif self.round > 200 and self.best_height >= self.EASY_CUTOFF and (self.cease_explore or (self.avg_height is not None
            and (max((self.best_abs_y - self.y) % self.grid_size, (self.best_abs_x - self.x) % self.grid_size) > 30 + 1 * (self.avg_height - flood_h)))):
            self.cease_explore = True
            raw_dx = (self.best_abs_x - self.x) % self.grid_size
            raw_dy = (self.best_abs_y - self.y) % self.grid_size
            dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
            dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
            movex, movey = sign(dx), sign(dy)

            if (movex == 0 and movey == 0) or abs(dx) + abs(dy) < 40 or flood_h > VIEWPOINT_AVERAGE_HEIGHT - 2 * self.safe_turns:
                movex == 0 and movey == 0
            elif movex == 0 or self.startedZigZagX:
                self.startedZigZagX = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movex = 1
                else:
                    movex = -1
            elif movey == 0 or self.startedZigZagY:
                self.startedZigZagY = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movey = 1
                else:
                    movey = -1

        elif self.round <= 400:
            # self.safe_turns = 1
            if (self.index % 3 == 0 or self.index % 5 == 1):
                self.dir_x, self.dir_y = self.diags[self.index % 2 + 2]
            # if saw_another_bot and (self.round - self.last_randomize_round > 10):
            #     self.dir_x, self.dir_y = random.choice(self.diags)
            #     self.last_randomize_round = self.round
            movex, movey = self.dir_x, self.dir_y
            if just_found_slope:
                movex, movey = sign(dx_local), sign(dy_local)
        elif self.turns_since_improve > -1 and self.best_height >= self.EASY_CUTOFF:
            raw_dx = (self.best_abs_x - self.x) % 512
            raw_dy = (self.best_abs_y - self.y) % 512
            dx = raw_dx - 512 if raw_dx > 256 else raw_dx
            dy = raw_dy - 512 if raw_dy > 256 else raw_dy
            movex, movey = sign(dx), sign(dy)
            if just_found_slope:
                movex, movey = sign(dx_local), sign(dy_local)
            
            if (movex == 0 and movey == 0) or abs(dx) + abs(dy) < 40 or flood_h > VIEWPOINT_AVERAGE_HEIGHT - 2 * self.safe_turns:
                movex == 0 and movey == 0
            elif movex == 0 or self.startedZigZagX:
                self.startedZigZagX = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movex = 1
                else:
                    movex = -1
            elif movey == 0 or self.startedZigZagY:
                self.startedZigZagY = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movey = 1
                else:
                    movey = -1
        else:
            movex, movey = self.dir_x, self.dir_y
            if just_found_slope:
                movex, movey = sign(dx_local), sign(dy_local)


        raw_dx = (self.best_abs_x - self.x) % 512
        raw_dy = (self.best_abs_y - self.y) % 512
        dx = raw_dx - 512 if raw_dx > 256 else raw_dx
        dy = raw_dy - 512 if raw_dy > 256 else raw_dy
        if self.scout_count > 0 and VIEWPOINT_AVERAGE_HEIGHT - flood_h > max(abs(self.best_abs_x - self.x), abs(self.best_abs_y - self.y)) + 3 * self.safe_turns and (
            self.scout or 650 >= self.round >= 400 and abs(dx) + abs(dy) < 8):
            self.scout = True
            self.scout_count -= 1
            movex, movey = self.dir_x, self.dir_y
        elif self.scout_count > 0 and (self.scout or self.round < 350 and (abs(dx) + abs(dy) < 8) or self.cease_explore):
            self.scout = True
            self.scout_count -= 0.2
            movex, movey = self.dir_x, self.dir_y
        if self.round == 399:
            self.scout = False
            self.scout_count = 10

        if self.x == self.best_abs_x and self.y == self.best_abs_y:
            self.offset_chosen = False  # reset so new offset will be chosen
        if (self.offset_chosen or self.x == self.best_abs_x and self.y == self.best_abs_y) and self.round < 550:
            # choose offset once at peak arrival
            if not self.offset_chosen and self.x == self.best_abs_x and self.y == self.best_abs_y:
                random.seed(self.index + self.round)
                max_r = 25 + (self.index % 11)
                min_r = 15
                while True:
                    dx = random.randint(-max_r, max_r)
                    dy = random.randint(-max_r, max_r)
                    if math.hypot(dx, dy) >= min_r:
                        break
                self.offset = (dx, dy)
                self.offset_chosen = True
            # move toward offset
            tx = (self.best_abs_x + self.offset[0]) % grid_size
            ty = (self.best_abs_y + self.offset[1]) % grid_size
            dx_off = (tx - self.x) % grid_size
            dy_off = (ty - self.y) % grid_size
            dx_signed = dx_off - grid_size if dx_off > grid_size//2 else dx_off
            dy_signed = dy_off - grid_size if dy_off > grid_size//2 else dy_off
            movex, movey = sign(dx_signed), sign(dy_signed)
            tx = (self.x + movex) % self.grid_size
            ty = (self.y + movey) % self.grid_size
            candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            if (tx, ty) in self.impassable:
                movex = sign(dxg)
                movey = sign(dyg)
            if flood_h + 4*self.safe_turns > center_h:
                movex = sign(dxg)
                movey = sign(dyg)
            if (dx_signed == 0 and dy_signed == 0):
                movex = sign(dxg)
                movey = sign(dyg)
                self.offset_chosen = False

        # ————————————————
        # 3) BUGNAV AROUND ANY “IMPASSABLE” DESTINATION
        tx = (self.x + movex) % self.grid_size
        ty = (self.y + movey) % self.grid_size
        candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        if (tx, ty) in self.impassable:
            # if first time bugging, mark old cell impassable and pick detour
            if not self.bugging:
                old_x, old_y = self.x, self.y
                self.impassable.add((old_x, old_y))
                candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                best = None
                best_score = None
                for dx, dy in candidates:
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    if (gx, gy) in self.impassable:
                        continue
                    score = (dx-movex)**2 + (dy-movey)**2
                    if best is None or score < best_score:
                        best = (dx, dy)
                        best_score = score
                if best is not None:
                    movex, movey = best
                    self.bugging = True
                    self.bug_dir = best
                else:
                    self.bug_dir = (-self.bug_dir[0], -self.bug_dir[1])
            else:
                movex, movey = self.bug_dir
                tx = (self.x + movex) % self.grid_size
                ty = (self.y + movey) % self.grid_size
                # old_x, old_y = self.x, self.y
                # self.impassable.add((old_x, old_y))
                candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                best = None
                best_score = None
                for dx, dy in candidates:
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    if (gx, gy) in self.impassable:
                        continue
                    score = (dx-movex)**2 + (dy-movey)**2
                    if best is None or score < best_score:
                        best = (dx, dy)
                        best_score = score
                if best is not None:
                    movex, movey = best
                    self.bugging = True
                    self.bug_dir = best
                else:
                    self.bug_dir = (-self.bug_dir[0], -self.bug_dir[1])
        else:
            self.bugging = False

        # ————————————————
        # 4) UPDATE POSITION
        self.x = (self.x + movex) % 512
        self.y = (self.y + movey) % 512

        # ————————————————
        # 5) ENCODE + BROADCAST
        raw_dx = (self.best_abs_x - self.x) % self.grid_size
        raw_dy = (self.best_abs_y - self.y) % self.grid_size
        adj_dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        adj_dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
        h_code  = min(int(self.best_height), 0x3FFF) & 0x3FFF
        dx_code = (adj_dx + 256) & 0x1FF
        dy_code = (adj_dy + 256) & 0x1FF
        m = (h_code << 18) | (dx_code << 9) | dy_code

        return movex, movey, m
    

    # MARK: MEDIUM MODE LOGIC
    def stepMedium(
        self,
        height: np.ndarray,
        neighbors: List[Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        

        VIEWPOINT_AVERAGE_HEIGHT = 0

        # ————————————————
        # 0) MARK “IMPASSABLE” TILES
        # flood height at end of this step will be ~ self.round+1,
        # so anything with terrain ≤ current_flood + safe_turns is unsafe.
        threshold = max(float(self.round + self.safe_turns), 400)
        H, W = height.shape
        for i in range(H):
            for j in range(W):
                h = float(height[i, j])
                VIEWPOINT_AVERAGE_HEIGHT += h
                if h <= threshold:
                    # compute the global coord of (i,j)
                    dx = i - view_radius
                    dy = j - view_radius
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    self.impassable.add((gx, gy))
        
        VIEWPOINT_AVERAGE_HEIGHT /= H * W


        center_h = float(height[view_radius, view_radius])
        self.sum_heights += VIEWPOINT_AVERAGE_HEIGHT
        self.count_heights += 1
        self.avg_height = self.sum_heights / self.count_heights

        # ————————————————
        # 1) local scan
        center_h = float(height[view_radius, view_radius])
        if self.best_height is None:
            self.best_height = center_h
            self.best_abs_x = self.x
            self.best_abs_y = self.y
            self.turns_since_improve = 0

        ind = np.unravel_index(np.argmax(height, axis=None), height.shape)
        local_h = height[ind]
        dx_local = ind[0] - view_radius
        dy_local = ind[1] - view_radius
        just_found_local = False
        if local_h > self.best_height:
            self.best_height = local_h
            self.best_abs_x = (self.x + dx_local) % 512
            self.best_abs_y = (self.y + dy_local) % 512
            self.turns_since_improve = 0
            just_found_local = True
            just_dx, just_dy = dx_local, dy_local
        else:
            self.turns_since_improve += 1

        # neighbor messages
        saw_another_bot = False
        for nx, ny, m in neighbors:
            if (nx == -self.dir_x and ny == -self.dir_y):
                continue

            h_code = (m >> 18) & 0x3FFF
            dx_code = (m >> 9) & 0x1FF
            dy_code = m & 0x1FF
            rel_dx = dx_code - 256
            rel_dy = dy_code - 256
            neigh_abs_x = (self.x + nx) % 512
            neigh_abs_y = (self.y + ny) % 512
            nbx = (neigh_abs_x + rel_dx) % 512
            nby = (neigh_abs_y + rel_dy) % 512
            if h_code > self.best_height:
                self.cease_explore = False
                self.best_height = float(h_code)
                self.best_abs_x = nbx
                self.best_abs_y = nby
                self.turns_since_improve = 0

            saw_another_bot = True


        raw_dx = (self.best_abs_x - self.x) % self.grid_size
        raw_dy = (self.best_abs_y - self.y) % self.grid_size
        dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
        dxg = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        dyg = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy

        # ————————————————
        # 2) ORIGINAL MOVEMENT DECISION
        self.round += 1
        flood_h = float(self.round)
        if self.round <= 70:
            movex, movey = self.dir_x, self.dir_y
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)
        elif self.round > 270 and self.best_height >= self.MEDIUM_CUTOFF and (self.cease_explore or (self.avg_height is not None
            and (max((self.best_abs_y - self.y) % self.grid_size, (self.best_abs_x - self.x) % self.grid_size) > 40 + 1.05 * (self.avg_height - flood_h)))):
            self.cease_explore = True
            raw_dx = (self.best_abs_x - self.x) % self.grid_size
            raw_dy = (self.best_abs_y - self.y) % self.grid_size
            dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
            dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
            movex, movey = sign(dx), sign(dy)

            if (movex == 0 and movey == 0) or abs(dx) + abs(dy) < 40 or flood_h > VIEWPOINT_AVERAGE_HEIGHT - 2 * self.safe_turns:
                movex == 0 and movey == 0
            elif movex == 0 or self.startedZigZagX:
                self.startedZigZagX = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movex = 1
                else:
                    movex = -1
            elif movey == 0 or self.startedZigZagY:
                self.startedZigZagY = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movey = 1
                else:
                    movey = -1

        elif self.round <= 420:
            # self.safe_turns = 1
            if (self.index % 3 == 0 or self.index % 5 == 1):
                self.dir_x, self.dir_y = self.diags[self.index % 2 + 2]
            # if saw_another_bot and (self.round - self.last_randomize_round > 10):
            #     self.dir_x, self.dir_y = random.choice(self.diags)
            #     self.last_randomize_round = self.round
            movex, movey = self.dir_x, self.dir_y
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)
        elif self.turns_since_improve > -1 and self.best_height >= self.MEDIUM_CUTOFF:
            raw_dx = (self.best_abs_x - self.x) % 512
            raw_dy = (self.best_abs_y - self.y) % 512
            dx = raw_dx - 512 if raw_dx > 256 else raw_dx
            dy = raw_dy - 512 if raw_dy > 256 else raw_dy
            movex, movey = sign(dx), sign(dy)
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)
            
            if (movex == 0 and movey == 0) or abs(dx) + abs(dy) < 40 or flood_h > VIEWPOINT_AVERAGE_HEIGHT - 2 * self.safe_turns:
                movex == 0 and movey == 0
            elif movex == 0 or self.startedZigZagX:
                self.startedZigZagX = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movex = 1
                else:
                    movex = -1
            elif movey == 0 or self.startedZigZagY:
                self.startedZigZagY = True
                self.zigzag -= 1
                if (self.zigzag < 0):
                    self.zigzagState = not self.zigzagState
                    self.zigzag = 20
                if self.zigzagState:
                    movey = 1
                else:
                    movey = -1
        else:
            movex, movey = self.dir_x, self.dir_y
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)


        raw_dx = (self.best_abs_x - self.x) % 512
        raw_dy = (self.best_abs_y - self.y) % 512
        dx = raw_dx - 512 if raw_dx > 256 else raw_dx
        dy = raw_dy - 512 if raw_dy > 256 else raw_dy
        # if self.scout_count > 0 and VIEWPOINT_AVERAGE_HEIGHT - flood_h > max(abs(self.best_abs_x - self.x), abs(self.best_abs_y - self.y)) + 3 * self.safe_turns and (
        #     self.scout or 600 >= self.round >= 352 and abs(dx) + abs(dy) < 8):
        #     self.scout = True
        #     self.scout_count -= 1
        #     if (self.scout_count == 0):
        #         self.scout_count = 30
        #         self.index += 1
        #         self.dir_x, self.dir_y = self.diagsSquare[self.index % 4]
        #     movex, movey = self.dir_x, self.dir_y
        # elif self.scout_count > 0 and (self.scout or self.round < 350 and (abs(dx) + abs(dy) < 8) or self.cease_explore):
        #     self.scout = True
        #     self.scout_count -= 0.2
        #     movex, movey = self.dir_x, self.dir_y
        # if self.round == 351:
        #     self.scout = False
        #     self.scout_count = 10
        
        if self.x == self.best_abs_x and self.y == self.best_abs_y:
            self.offset_chosen = False  # reset so new offset will be chosen
        if (self.offset_chosen or self.x == self.best_abs_x and self.y == self.best_abs_y) and self.round < 700:
            # choose offset once at peak arrival
            if not self.offset_chosen and self.x == self.best_abs_x and self.y == self.best_abs_y:
                random.seed(self.index + self.round)
                max_r = 23 + (self.index % 9)
                min_r = 10
                if (self.round < 600):
                    max_r += 80
                    min_r += 10
                elif (self.round < 650):
                    max_r += 10
                    min_r += 5
                while True:
                    dx = random.randint(-max_r, max_r)
                    dy = random.randint(-max_r, max_r)
                    if math.hypot(dx, dy) >= min_r:
                        break
                self.offset = (dx, dy)
                self.offset_chosen = True
            # move toward offset
            tx = (self.best_abs_x + self.offset[0]) % grid_size
            ty = (self.best_abs_y + self.offset[1]) % grid_size
            dx_off = (tx - self.x) % grid_size
            dy_off = (ty - self.y) % grid_size
            dx_signed = dx_off - grid_size if dx_off > grid_size//2 else dx_off
            dy_signed = dy_off - grid_size if dy_off > grid_size//2 else dy_off
            movex, movey = sign(dx_signed), sign(dy_signed)
            tx = (self.x + movex) % self.grid_size
            ty = (self.y + movey) % self.grid_size
            candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            if (tx, ty) in self.impassable:
                movex = sign(dxg)
                movey = sign(dyg)
            # if flood threatens current pos, retreat to peak
            if flood_h + 4*self.safe_turns > center_h:
                movex = sign(dxg)
                movey = sign(dyg)
            if (dx_signed == 0 and dy_signed == 0):
                movex = sign(dxg)
                movey = sign(dyg)
                self.offset_chosen = False

        # ————————————————
        # 3) BUGNAV AROUND ANY “IMPASSABLE” DESTINATION
        tx = (self.x + movex) % self.grid_size
        ty = (self.y + movey) % self.grid_size
        candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        if (tx, ty) in self.impassable:
            # if first time bugging, mark old cell impassable and pick detour
            if not self.bugging:
                old_x, old_y = self.x, self.y
                self.impassable.add((old_x, old_y))
                candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                best = None
                best_score = None
                for dx, dy in candidates:
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    if (gx, gy) in self.impassable:
                        continue
                    score = (dx-movex)**2 + (dy-movey)**2
                    if best is None or score < best_score:
                        best = (dx, dy)
                        best_score = score
                if best is not None:
                    movex, movey = best
                    self.bugging = True
                    self.bug_dir = best
                else:
                    self.bug_dir = (-self.bug_dir[0], -self.bug_dir[1])
            else:
                movex, movey = self.bug_dir
                tx = (self.x + movex) % self.grid_size
                ty = (self.y + movey) % self.grid_size
                # old_x, old_y = self.x, self.y
                # self.impassable.add((old_x, old_y))
                candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                best = None
                best_score = None
                for dx, dy in candidates:
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    if (gx, gy) in self.impassable:
                        continue
                    score = (dx-movex)**2 + (dy-movey)**2
                    if best is None or score < best_score:
                        best = (dx, dy)
                        best_score = score
                if best is not None:
                    movex, movey = best
                    self.bugging = True
                    self.bug_dir = best
                else:
                    self.bug_dir = (-self.bug_dir[0], -self.bug_dir[1])
        else:
            self.bugging = False

        # ————————————————
        # 4) UPDATE POSITION
        self.x = (self.x + movex) % 512
        self.y = (self.y + movey) % 512

        # ————————————————
        # 5) ENCODE + BROADCAST
        raw_dx = (self.best_abs_x - self.x) % self.grid_size
        raw_dy = (self.best_abs_y - self.y) % self.grid_size
        adj_dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        adj_dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
        h_code  = min(int(self.best_height), 0x3FFF) & 0x3FFF
        dx_code = (adj_dx + 256) & 0x1FF
        dy_code = (adj_dy + 256) & 0x1FF
        m = (h_code << 18) | (dx_code << 9) | dy_code

        if (movex == 0 and movey == 0) and self.round < 150:
            movex, movey = self.dir_x, self.dir_y

        return movex, movey, m
    

    # MARK: HARD MODE LOGIC
    def detect_barriers(self, height: np.ndarray) -> None:
        # scan for horizontal cracks of thickness 3
        H, W = height.shape
        for i in range(H - 2):
            band = height[:, i:i+3]
            mean_low = float(np.mean(band))
            std_low = float(np.std(band))
            if std_low < BARRIER_EPS:
                if 0 < i < H-3:
                    mean_above = float(np.mean(height[:, i-1]))
                    mean_below = float(np.mean(height[:, i+3]))
                    if mean_low + BARRIER_DELTA < min(mean_above, mean_below):
                        # record global y coordinate of this crack center
                        dy = i + 1 - view_radius
                        gy = (self.y + dy) % self.grid_size
                        self.barriers.add(gy)
                        self.barrier_heights.setdefault(gy, mean_low)
    
    def stepHard(
        self,
        height: np.ndarray,
        neighbors: List[Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        
        center_h = float(height[view_radius, view_radius])
        self.sum_heights += center_h
        self.count_heights += 1
        self.avg_height = self.sum_heights / self.count_heights

        VIEWPOINT_AVERAGE_HEIGHT = 0
        self.detect_barriers(height)

        # ————————————————
        # 0) MARK “IMPASSABLE” TILES
        # flood height at end of this step will be ~ self.round+1,
        # so anything with terrain ≤ current_flood + safe_turns is unsafe.
        threshold = float(self.round + self.safe_turns)
        H, W = height.shape
        for i in range(H):
            for j in range(W):
                h = float(height[i, j])
                VIEWPOINT_AVERAGE_HEIGHT += h
                if h <= threshold:
                    dx = i - view_radius
                    dy = j - view_radius
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    self.impassable.add((gx, gy))
        
        VIEWPOINT_AVERAGE_HEIGHT /= H * W

        # ————————————————
        # 1) local scan
        center_h = float(height[view_radius, view_radius])
        if self.best_height is None:
            self.best_height = center_h
            self.best_abs_x = self.x
            self.best_abs_y = self.y
            self.turns_since_improve = 0

        ind = np.unravel_index(np.argmax(height, axis=None), height.shape)
        local_h = float(height[ind])
        dx_local = ind[0] - view_radius
        dy_local = ind[1] - view_radius
        just_found_local = False
        if local_h > self.best_height:
            # Update best
            self.best_height = local_h
            self.best_abs_x = (self.x + dx_local) % 512
            self.best_abs_y = (self.y + dy_local) % 512
            self.turns_since_improve = 0
            just_found_local = True
            just_dx, just_dy = dx_local, dy_local
        else:
            self.turns_since_improve += 1

        # neighbor messages
        saw_another_bot = False
        for nx, ny, m in neighbors:
            if (nx == -self.dir_x and ny == -self.dir_y):
                continue

            h_code = (m >> 18) & 0x3FFF
            dx_code = (m >> 9) & 0x1FF
            dy_code = m & 0x1FF
            rel_dx = dx_code - 256
            rel_dy = dy_code - 256
            neigh_abs_x = (self.x + nx) % 512
            neigh_abs_y = (self.y + ny) % 512
            nbx = (neigh_abs_x + rel_dx) % 512
            nby = (neigh_abs_y + rel_dy) % 512
            if h_code > self.best_height:
                self.best_height = float(h_code)
                self.best_abs_x = nbx
                self.best_abs_y = nby
                self.turns_since_improve = 0

            saw_another_bot = True


        raw_dx = (self.best_abs_x - self.x) % self.grid_size
        raw_dy = (self.best_abs_y - self.y) % self.grid_size
        dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
        dxg = raw_dx - grid_size if raw_dx > grid_size//2 else raw_dx
        dyg = raw_dy - grid_size if raw_dy > grid_size//2 else raw_dy

        # ————————————————
        # 2) ORIGINAL MOVEMENT DECISION
        self.round += 1
        flood_h = float(self.round)
        if self.round <= 50:
            movex, movey = self.dir_x, self.dir_y
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)
        elif self.round >= 200 and self.best_height >= self.HARD_CUTOFF and (self.avg_height is not None
            and 1.2 * (abs(dx) + abs(dy)) > (self.avg_height - flood_h)):
            raw_dx = (self.best_abs_x - self.x) % self.grid_size
            raw_dy = (self.best_abs_y - self.y) % self.grid_size
            dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
            dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
            dxg = raw_dx - grid_size if raw_dx > grid_size//2 else raw_dx
            dyg = raw_dy - grid_size if raw_dy > grid_size//2 else raw_dy
            movex, movey = sign(dx), sign(dy)
            
            # plan route
            direction = 1 if dyg > 0 else -1
            def in_between(cur, end, by):
                # works on torus but approximate linear segment ig
                return direction*(by - cur) > 0 and direction*(by - end) <= 0

            # find earliest barrier that will activate before arrival
            target_y = self.best_abs_y
            for by in sorted(self.barriers, key=lambda yy: direction*(yy - self.y)):
                if in_between(self.y, target_y, by):
                    dy_dist = abs((by - self.y + grid_size) % grid_size if direction>0 else (self.y - by + grid_size)%grid_size)
                    arrival_round = self.round + dy_dist
                    activation_round = int(self.barrier_heights[by])
                    if activation_round <= arrival_round:
                        if by in self.openings and self.openings[by]:
                            opens = list(self.openings[by])
                            best_open = min(opens, key=lambda ox: abs((ox - self.x + grid_size)%grid_size if ( (ox-self.x+grid_size)%grid_size )<=grid_size//2 else -((ox-self.x+grid_size)%grid_size)))
                            movex = sign((best_open - self.x + grid_size)%grid_size - (grid_size if (best_open - self.x + grid_size)%grid_size>grid_size//2 else 0))
                            movey = sign(by - self.y if direction>0 else -(self.y-by))
                            break
                        movex, movey = sign(dxg), sign(dyg)
                        break
            else:
                movex, movey = sign(dxg), sign(dyg)

        elif self.round <= 400:
            # self.safe_turns = 1
            if (self.index % 3 == 0 or self.index % 5 == 1):
                self.dir_x, self.dir_y = self.diags[self.index % 2 + 2]
            # if saw_another_bot and (self.round - self.last_randomize_round > 10):
            #     self.dir_x, self.dir_y = random.choice(self.diags)
            #     self.last_randomize_round = self.round
            movex, movey = self.dir_x, self.dir_y
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)
        elif self.best_height >= self.HARD_CUTOFF and self.turns_since_improve > -1:
            raw_dx = (self.best_abs_x - self.x) % 512
            raw_dy = (self.best_abs_y - self.y) % 512
            dx = raw_dx - 512 if raw_dx > 256 else raw_dx
            dy = raw_dy - 512 if raw_dy > 256 else raw_dy
            dxg = raw_dx - grid_size if raw_dx > grid_size//2 else raw_dx
            dyg = raw_dy - grid_size if raw_dy > grid_size//2 else raw_dy
            movex, movey = sign(dx), sign(dy)

            # plan route
            direction = 1 if dyg > 0 else -1
            def in_between(cur, end, by):
                # works on torus but approximate linear segment ig
                return direction*(by - cur) > 0 and direction*(by - end) <= 0

            target_y = self.best_abs_y
            for by in sorted(self.barriers, key=lambda yy: direction*(yy - self.y)):
                if in_between(self.y, target_y, by):
                    dy_dist = abs((by - self.y + grid_size) % grid_size if direction>0 else (self.y - by + grid_size)%grid_size)
                    arrival_round = self.round + dy_dist
                    activation_round = int(self.barrier_heights[by])
                    if activation_round <= arrival_round:
                        if by in self.openings and self.openings[by]:
                            opens = list(self.openings[by])
                            best_open = min(opens, key=lambda ox: abs((ox - self.x + grid_size)%grid_size if ( (ox-self.x+grid_size)%grid_size )<=grid_size//2 else -((ox-self.x+grid_size)%grid_size)))
                            movex = sign((best_open - self.x + grid_size)%grid_size - (grid_size if (best_open - self.x + grid_size)%grid_size>grid_size//2 else 0))
                            movey = sign(by - self.y if direction>0 else -(self.y-by))
                            break
                        movex, movey = sign(dxg), sign(dyg)
                        break
            else:
                movex, movey = sign(dxg), sign(dyg)

            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)
        else:
            movex, movey = self.dir_x, self.dir_y
            if just_found_local:
                movex, movey = sign(just_dx), sign(just_dy)


        # if self.scout_count > 0 and VIEWPOINT_AVERAGE_HEIGHT - flood_h > max(abs(self.best_abs_x - self.x), abs(self.best_abs_y - self.y)) + 3 * self.safe_turns and (
        #     self.scout or self.round >= 400 and abs(self.best_abs_x - self.x) + abs(self.best_abs_y - self.y) < 8):
        #     self.scout = True
        #     self.scout_count -= 1
        #     movex, movey = self.dir_x, self.dir_y
        if self.x == self.best_abs_x and self.y == self.best_abs_y:
            self.offset_chosen = False  # reset so new offset will be chosen
        if (self.offset_chosen or self.x == self.best_abs_x and self.y == self.best_abs_y) and self.round < 850:
            # choose offset once at peak arrival
            if not self.offset_chosen and self.x == self.best_abs_x and self.y == self.best_abs_y:
                random.seed(self.index + self.round)
                max_r = 20 + (self.index % 11)
                min_r = 12
                r2 = random.uniform(min_r*min_r, max_r*max_r)
                r = math.sqrt(r2)
                theta = random.random() * 2 * math.pi
                dx = int(round(r * math.cos(theta)))
                dy = int(round(r * math.sin(theta)))
                self.offset = (dx, dy)
                self.offset_chosen = True

            tx = (self.best_abs_x + self.offset[0]) % grid_size
            ty = (self.best_abs_y + self.offset[1]) % grid_size
            dx_off = (tx - self.x) % grid_size
            dy_off = (ty - self.y) % grid_size
            dx_signed = dx_off - grid_size if dx_off > grid_size//2 else dx_off
            dy_signed = dy_off - grid_size if dy_off > grid_size//2 else dy_off

            # if too close to any barrier, go back to peak and pick new offset
            if any(abs((self.y - by + grid_size) % grid_size if (self.y - by + grid_size)%grid_size<=grid_size//2 else abs((by - self.y + grid_size)%grid_size)) <= 2 for by in self.barriers):
                movex = sign(dxg)
                movey = sign(dyg)
                self.offset_chosen = False
            else:
                movex, movey = sign(dx_signed), sign(dy_signed)
                if flood_h + 4*self.safe_turns > center_h:
                    movex = sign(dxg)
                    movey = sign(dyg)
                if (dx_signed == 0 and dy_signed == 0):
                    movex = sign(dxg)
                    movey = sign(dyg)
                    self.offset_chosen = False

        # ————————————————
        # 3) BUGNAV AROUND ANY “IMPASSABLE” DESTINATION
        tx = (self.x + movex) % self.grid_size
        ty = (self.y + movey) % self.grid_size
        candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        if (tx, ty) in self.impassable:
            # if first time bugging, mark old cell impassable and pick detour
            if not self.bugging:
                old_x, old_y = self.x, self.y
                self.impassable.add((old_x, old_y))
                candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                best = None
                best_score = None
                for dx, dy in candidates:
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    if (gx, gy) in self.impassable:
                        continue
                    score = (dx-movex)**2 + (dy-movey)**2
                    if best is None or score < best_score:
                        best = (dx, dy)
                        best_score = score
                if best is not None:
                    movex, movey = best
                    self.bugging = True
                    self.bug_dir = best
                else:
                    self.bug_dir = (-self.bug_dir[0], -self.bug_dir[1])
            else:
                movex, movey = self.bug_dir
                tx = (self.x + movex) % self.grid_size
                ty = (self.y + movey) % self.grid_size
                # old_x, old_y = self.x, self.y
                # self.impassable.add((old_x, old_y))
                candidates = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                best = None
                best_score = None
                for dx, dy in candidates:
                    gx = (self.x + dx) % self.grid_size
                    gy = (self.y + dy) % self.grid_size
                    if (gx, gy) in self.impassable:
                        continue
                    score = (dx-movex)**2 + (dy-movey)**2
                    if best is None or score < best_score:
                        best = (dx, dy)
                        best_score = score
                if best is not None:
                    movex, movey = best
                    self.bugging = True
                    self.bug_dir = best
                else:
                    self.bug_dir = (-self.bug_dir[0], -self.bug_dir[1])
        else:
            self.bugging = False

        # ————————————————
        # 4) UPDATE POSITION
        self.x = (self.x + movex) % 512
        self.y = (self.y + movey) % 512
        for by, bh in self.barrier_heights.items():
            if self.y == by and center_h > bh:
                self.openings.setdefault(by, set()).add(self.x)


        # ————————————————
        # 5) ENCODE + BROADCAST
        raw_dx = (self.best_abs_x - self.x) % self.grid_size
        raw_dy = (self.best_abs_y - self.y) % self.grid_size
        adj_dx = raw_dx - self.grid_size if raw_dx > self.grid_size//2 else raw_dx
        adj_dy = raw_dy - self.grid_size if raw_dy > self.grid_size//2 else raw_dy
        h_code  = min(int(self.best_height), 0x3FFF) & 0x3FFF
        dx_code = (adj_dx + 256) & 0x1FF
        dy_code = (adj_dy + 256) & 0x1FF
        m = (h_code << 18) | (dx_code << 9) | dy_code

        return movex, movey, m
    