import math
from typing import Dict, List, Tuple

import pygame

from emergency_lights.light_controller import LightController


class PygameIntersection:
    """
    Pygame-based visualization of a four-way intersection with emergency lights and ambulances.
    """

    def __init__(self, controller: LightController, manual_spawns: bool = True) -> None:
        pygame.init()
        pygame.display.set_caption("Smart Traffic Ambulance")
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.controller = controller
        self.manual_spawns = manual_spawns
        self.font = pygame.font.SysFont(None, 28)
        self.banner_font = pygame.font.SysFont(None, 36)
        self.ambulances: Dict[str, List[Dict[str, float]]] = {"North": [], "South": [], "East": [], "West": []}
        self.running = True

    def _spawn_ambulance(self, road: str) -> None:
        if road == "North":
            self.ambulances[road].append({"x": 400.0, "y": -60.0, "vx": 0.0, "vy": 5.0})
        elif road == "South":
            self.ambulances[road].append({"x": 400.0, "y": 860.0, "vx": 0.0, "vy": -5.0})
        elif road == "East":
            self.ambulances[road].append({"x": 860.0, "y": 400.0, "vx": -5.0, "vy": 0.0})
        elif road == "West":
            self.ambulances[road].append({"x": -60.0, "y": 400.0, "vx": 5.0, "vy": 0.0})
        self.controller.turn_on(road)

    def _update_ambulances(self) -> None:
        to_remove: List[Tuple[str, int]] = []
        for road, items in self.ambulances.items():
            for i, a in enumerate(items):
                a["x"] += a["vx"]
                a["y"] += a["vy"]
                if a["x"] < -100 or a["x"] > 900 or a["y"] < -100 or a["y"] > 900:
                    to_remove.append((road, i))
        for road, idx in reversed(to_remove):
            self.ambulances[road].pop(idx)
            self.controller.turn_off(road)

    def _draw_intersection(self) -> None:
        self.screen.fill((34, 34, 34))
        # Draw roads
        pygame.draw.rect(self.screen, (60, 60, 60), pygame.Rect(300, 0, 200, 800))
        pygame.draw.rect(self.screen, (60, 60, 60), pygame.Rect(0, 300, 800, 200))
        
        # Position mapping for lights and labels
        positions = {
            "North": {"light": (400, 80), "label": (370, 30)},
            "South": {"light": (400, 720), "label": (370, 750)},
            "East": {"light": (720, 400), "label": (750, 370)},
            "West": {"light": (80, 400), "label": (10, 370)},
        }

        # Draw emergency lights for configured roads
        for road in self.controller.get_roads():
            state, color = self.controller.get_visual_state(road)
            pos_data = positions.get(road)
            if pos_data:
                cx, cy = pos_data["light"]
                lx, ly = pos_data["label"]
                
                # Draw outer glow if ON
                if state == "ON":
                    glow_color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50))
                    pygame.draw.circle(self.screen, glow_color, (cx, cy), 28)
                
                # Draw light circle
                pygame.draw.circle(self.screen, color, (cx, cy), 20)
                
                # Draw label
                lbl = self.font.render(road, True, (230, 230, 230))
                self.screen.blit(lbl, (lx, ly))

        # Draw ambulances
        for road, items in self.ambulances.items():
            for a in items:
                rect = pygame.Rect(int(a["x"]), int(a["y"]), 60, 30)
                pygame.draw.rect(self.screen, (240, 240, 240), rect)
                # Draw red cross
                cx, cy = rect.center
                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(cx - 3, rect.y + 5, 6, rect.h - 10))
                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(rect.x + 10, cy - 3, rect.w - 20, 6))

        # Draw emergency banner
        active = self.controller.get_active_roads()
        if active:
            text = "⚠️ EMERGENCY — " + ", ".join(active)
            banner = self.banner_font.render(text, True, (255, 230, 80))
            _, bh = banner.get_size()
            pygame.draw.rect(self.screen, (60, 0, 0), pygame.Rect(0, 0, 800, bh + 10))
            self.screen.blit(banner, (10, 5))

        # UI Hints
        if self.manual_spawns:
            hint_text = "Keys: N/S/E/W to spawn, R to reset"
        else:
            hint_text = "Camera-only mode: manual spawns disabled"
        
        hint = self.font.render(hint_text, True, (180, 180, 180))
        self.screen.blit(hint, (10, 770))

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.controller.reset()
                    elif self.manual_spawns:
                        if event.key == pygame.K_n:
                            self._spawn_ambulance("North")
                        elif event.key == pygame.K_s:
                            self._spawn_ambulance("South")
                        elif event.key == pygame.K_e:
                            self._spawn_ambulance("East")
                        elif event.key == pygame.K_w:
                            self._spawn_ambulance("West")
            self._update_ambulances()
            self._draw_intersection()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()
