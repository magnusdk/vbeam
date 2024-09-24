"""A datastructure for representing a transducer probe excluding 
the position, orientation, etc.
"""

from typing import Callable, Optional
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import distance

identity_fn = lambda x: x  # Just return value as-is


@traceable_dataclass(("ROC", "rx_aperture_length_s", "tx_aperture_length_s"))
class ProbeGeometry:
    """A container for probe parameters geometries.

    """
    ROC: tuple[float, float] # (azimuth, elevation)
    rx_aperture_length_s: tuple[float, float] = None # (width, height)
    tx_aperture_length_s: tuple[float, float] = None # (width, height)
    def __postinit__(self):           
        if type(self.ROC) is not tuple or len(self.ROC) != 2:
            self.ROC = (self.ROC, self.ROC)
    def __getitem__(self, *args) -> "ProbeGeometry":
        _maybe_getitem = lambda attr: attr.__getitem__(*args) if attr is not None else None
        return self.ProbeGeometry(
            _maybe_getitem(self.ROC),
            _maybe_getitem(self.rx_aperture_length_s),
            _maybe_getitem(self.tx_aperture_length_s),
        )

    
    @property
    def curvature_center(self):
        return (np.array([0.0,0.0,-self.ROC[0]]), np.array([0.0,0.0,-self.ROC[1]]) ) 
    
    @property
    def get_sender_normal(self):
        vector = self.sender_position - self.curvature_center
        return vector / distance(vector)
    
    @property
    def get_receiver_normal(self):
        vector = self.receiver_position - self.curvature_center
        return vector / distance(vector)
    
    def get_theta(self,position):
        return np.arctan2(position[0],self.ROC[0]+position[2])
    
    def get_phi(self,position):
        return np.arctan2(position[1],self.ROC[1]+position[2])
    
    def get_surface_length(self, position):
        theta = self.get_theta(position=position)
        phi = self.get_phi(position=position)
        return (self.ROC*theta, self.ROC*phi)
    
    def cart2surface(self, position):
        return (self.ROC[0]*self.get_theta(position=position), self.ROC[1]*self.get_phi(position=position))
    
    def surface2cart(self,position_s):
        return np.array((
            self.ROC[0]*np.sin(position_s[0]/self.ROC[0]),
            self.ROC[1]*np.sin(position_s[1]/self.ROC[1]),
            np.where(self.ROC[0]-self.ROC[0]*np.cos(position_s[0]/self.ROC[0])>self.ROC[1]-self.ROC[1]*np.cos(position_s[1]/self.ROC[1]),
                     self.ROC[0]-self.ROC[0]*np.cos(position_s[0]/self.ROC[0]),
                     self.ROC[1]-self.ROC[1]*np.cos(position_s[1]/self.ROC[1])
                     )
        ))
    
    @property
    def get_rx_aperture_borders(self):
        left = self.surface2cart((-self.rx_aperture_length_s[0]/2,0.0))
        right = self.surface2cart((self.tx_aperture_length_s[0]/2,0.0))
        down = self.surface2cart((0.0, -self.tx_aperture_length_s[1]/2))
        up = self.surface2cart((0.0, self.tx_aperture_length_s[1]/2))
        return (left, right, up, down)
        return [
                np.array([self.ROC[0]*np.sin(self.rx_aperture_length_s[0]/2/self.ROC[0]),0.0,self.ROC[0]-self.ROC[0]*np.cos(self.rx_aperture_length_s[0]/2/self.ROC[0])]),
                np.array([-self.ROC[0]*np.sin(self.rx_aperture_length_s[0]/2/self.ROC[0]),0.0,self.ROC[0]-self.ROC[0]*np.cos(self.rx_aperture_length_s[0]/2/self.ROC[0])]),
                np.array([0.0,self.ROC[1]*np.sin(self.rx_aperture_length_s[1]/2/self.ROC[1]),self.ROC[1]-self.ROC[1]*np.cos(self.rx_aperture_length_s[1]/2/self.ROC[1])]),
                np.array([0.0,-self.ROC[1]*np.sin(self.rx_aperture_length_s[1]/2/self.ROC[1]),self.ROC-self.ROC[1]*np.cos(self.rx_aperture_length_s[1]/2/self.ROC[1])])
        ]

    def get_tx_aperture_borders(self, sender):
        left = self.surface2cart((self.cart2surface(position=sender)[0] - self.tx_aperture_length_s[0]/2,0.0))
        right = self.surface2cart((self.cart2surface(position=sender)[0] + self.tx_aperture_length_s[0]/2,0.0))
        down = self.surface2cart((0.0, self.cart2surface(position=sender)[1] - self.tx_aperture_length_s[1]/2))
        up = self.surface2cart((0.0, self.cart2surface(position=sender)[1] + self.tx_aperture_length_s[1]/2))
        return (left, right, up, down)
    
    def set_rx_aperture_length(self,min_x,max_x, min_y, max_y):
        width_s = np.arcsin(max_x/self.ROC[0])*self.ROC[0]-np.arcsin(min_x/self.ROC[0])*self.ROC[0]
        height_s = np.arcsin(max_y/self.ROC[1])*self.ROC[1]-np.arcsin(min_y/self.ROC[1])*self.ROC[1]
        self.rx_aperture_length_s = (width_s, height_s)

    def set_tx_aperture_length(self,min_x,max_x, min_y, max_y):
        width_s = np.arcsin(max_x/self.ROC[0])*self.ROC[0]-np.arcsin(min_x/self.ROC[0])*self.ROC[0]
        height_s = np.arcsin(max_y/self.ROC[1])*self.ROC[1]-np.arcsin(min_y/self.ROC[1])*self.ROC[1]
        self.tx_aperture_length_s = (width_s, height_s)

    def with_updates_to(
        self,
        *,
        ROC: Callable[[float], float] = identity_fn,
        rx_aperture_length_s: Callable[tuple[float,float],float] = identity_fn,
        tx_aperture_length_s: Callable[tuple[float,float],float] = identity_fn,
    ) -> "ProbeGeometry":
        return ProbeGeometry(
            ROC=ROC(self.ROC) if callable(ROC) else ROC,
            rx_aperture_length_s=rx_aperture_length_s(self.rx_aperture_length_s) if callable(rx_aperture_length_s) else rx_aperture_length_s,
            tx_aperture_length_s=tx_aperture_length_s(self.tx_aperture_length_s) if callable(tx_aperture_length_s) else tx_aperture_length_s,
        )

    def copy(self) -> "ProbeGeometry":
        return self.with_updates_to()
