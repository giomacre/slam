from decorators import ddict


camera_params = ddict(
    fx=517.3,
    fy=520.9,
    cx=318.6,
    cy=255.3,
    k1=0.2624,
    k2=-0.9531,
    p1=-0.0054,
    p2=0.0026,
    k3=1.1633,
    alpha=1.0,
)

frontend_params = ddict(
    n_features=350,
    kf_threshold=0.75,
)
