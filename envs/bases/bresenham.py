import numpy as np


class bresenham:
    def __init__(self, geo_grid_data_):
        self.geo_grid_data = geo_grid_data_

    def can_see_eachother(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        _return_value = True
        while True:
            _rgb_value = self.geo_grid_data.get_grid_RGB_property(x0, y0)
            if _rgb_value[1] > 10 or _rgb_value[0] > 10:  # G has value
                _return_value = False
                break
            # yield (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return _return_value

