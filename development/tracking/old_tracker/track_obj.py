class track:
    "Clase definida para trabajar con tracks como objetos así facilito cosas"
    nex_id = 0

    def __init__(self):
        self.id = track.nex_id
        self.points = []
        self.distances = []
        track.nex_id += 1

    def add_point(self, coordinates):
        self.points.add(coordinates)




