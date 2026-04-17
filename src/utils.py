class IDMapper:
    def __init__(self, ids):
        self.unique_ids = sorted(set(ids))
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.unique_ids)}
        self.idx_to_id = {i: id_ for i, id_ in enumerate(self.unique_ids)}

    def to_idx(self, ids):
        return ids.map(self.id_to_idx)

    @property
    def size(self):
        return len(self.unique_ids)