import pygame as pg
import numpy as np
import math
import numba
import taichi as ti


# ============== settings ==============
RES = WIDTH, HEIGHT = 800, 450
offset = np.array([1.3 * WIDTH, HEIGHT]) // 2
max_iter = 30
zoom = 2.2 / HEIGHT

# ============== texture ==============
texture = pg.image.load('images/texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture)


class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((WIDTH, HEIGHT, 3), [0, 0, 0],
                                    dtype=np.uint8)
        self.x = np.linspace(0, WIDTH, num=WIDTH, dtype=np.float32)
        self.y = np.linspace(0, HEIGHT, num=HEIGHT, dtype=np.float32)

        # ======= Control Settings =======
        self.vel = 0.01
        self.zoom, self.scale = 2.2 / HEIGHT, 0.993
        self.increment = [0, 0]
        self.max_iter, self.max_iter_limit = 30, 5500

        print(ti.Vector([0.0, 0.0]))

        # ======= Time =======
        self.app_speed = 1 / 4000
        self.prev_time = pg.time.get_ticks()

    def delta_time(self):
        time_now = pg.time.get_ticks() - self.prev_time
        self.prev_time = time_now
        return time_now * self.app_speed

    # ============== Numba ==============
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(screen_array, max_local_iter, local_zoom, dx, dy):

        for x in numba.prange(WIDTH):
            for y in range(HEIGHT):
                c = (((x - offset[0]) * local_zoom) - dx) + 1j * (
                        ((y - offset[1]) * local_zoom) - dy)
                z = 0
                num_iter = 0
                for i in range(max_local_iter):
                    z = z ** 2 + c
                    if z.real ** 2 + z.imag ** 2 > 4.0:
                        break
                    num_iter += 1
                # ======= gradient =======
                col = int(texture_size * num_iter / max_local_iter)
                screen_array[x, y] = texture_array[col, col]

                # ======= white and black=======
                # col = int(255 * num_iter / max_iter)
                # screen_array[x, y] = (col, col, col)
        return screen_array

    def control(self):
        pressed_key = pg.key.get_pressed()
        dt = self.delta_time()

        if pressed_key[pg.K_a]:
            self.increment[0] += self.vel * dt
        if pressed_key[pg.K_d]:
            self.increment[0] -= self.vel * dt
        if pressed_key[pg.K_w]:
            self.increment[1] += self.vel * dt
        if pressed_key[pg.K_s]:
            self.increment[1] -= self.vel * dt

        if pressed_key[pg.K_UP] or pressed_key[pg.K_DOWN]:
            inv_scale = 2 - self.scale
            if pressed_key[pg.K_UP]:
                self.zoom *= self.scale
                self.vel *= self.scale
            if pressed_key[pg.K_DOWN]:
                self.zoom *= inv_scale
                self.vel *= inv_scale

        if pressed_key[pg.K_LEFT]:
            self.max_iter -= 1
        if pressed_key[pg.K_RIGHT]:
            self.max_iter += 1
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)

    def update(self):
        self.control()
        self.screen_array = self.render(self.screen_array, self.max_iter, self.zoom,
                                        self.increment[0],
                                        self.increment[1])

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)
        pg.display.flip()

    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self):
        self.screen = pg.display.set_mode(RES, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps()}')


if __name__ == '__main__':
    app = App()
    app.run()
