# coords_transformer.py

class CoordinateTransformer:
    def __init__(self, img_width, img_height, map_width, map_height):
        """
        Sets up the coordinate transformation between image pixels (u, v)
        and physical robot scanner coordinate space (x, y) in mm.
        """
        self.img_w = float(img_width)
        self.img_h = float(img_height)
        self.map_w = float(map_width)
        self.map_h = float(map_height)

    def pixel_to_physics(self, u, v):
        """
        Maps pixel coordinate (u, v) to physical coordinate (x, y).
        Note: Image origin is top-left (v increases downwards).
        Physical origin is bottom-left (y increases upwards).
        """
        x = (u / self.img_w) * self.map_w
        y = (1.0 - (v / self.img_h)) * self.map_h
        return x, y

    def physics_to_pixel(self, x, y):
        """
        Maps physical coordinate (x, y) back to pixel coordinate (u, v).
        """
        u = (x / self.map_w) * self.img_w
        v = (1.0 - (y / self.map_h)) * self.img_h
        return u, v

    def pixel_box_to_physics_rect(self, u_min, v_min, u_max, v_max):
        """
        Maps a pixel obstacle bounding box [u_min, v_min, u_max, v_max]
        to a physical coordinate rectangle [x_min, y_min, x_max, y_max].
        """
        # Convert corners
        x_min = (u_min / self.img_w) * self.map_w
        x_max = (u_max / self.img_w) * self.map_w
        
        # Because Y is inverted, the maximum pixel v corresponds to the minimum physical y
        y_min = (1.0 - (v_max / self.img_h)) * self.map_h
        y_max = (1.0 - (v_min / self.img_h)) * self.map_h
        
        return [x_min, y_min, x_max, y_max]
