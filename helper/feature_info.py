

class FeatureINFO(object):
    def __init__(self, feature_names, discrete_cols, feature_dims):
        self.type_info = dict(zip(feature_names, ['continuous']*len(feature_names)))
        for name in discrete_cols:
            self.type_info[name] = 'discrete'
        self.dim_info = dict(zip(feature_names, feature_dims))

        positions = []
        cur_position = 0
        for i in range(len(feature_dims)):
            start = cur_position
            end = start + feature_dims[i]
            positions.append([item for item in range(start, end)])
            cur_position += feature_dims[i]
        self.pos_info = dict(zip(feature_names, positions))

    def get_position_by_name(self, name, sort=True):
        if isinstance(name, list):
            idx = []
            for n in name:
                idx.extend(self.pos_info[n])

            idx = sorted(idx) if sort else idx
        else:
            idx = self.pos_info[name]

        return idx