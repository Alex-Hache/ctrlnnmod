import numpy as np
from typing import Union, List, Tuple
from typeguard import typechecked
from collections.abc import Iterable, Sequence
from texttable import Texttable


def merge(*datasets):
    # TODO 
    pass


@typechecked
class DatasetIterator(Sequence):
    def __init__(self, mother):
        self._enable = len(mother) == 1
        self.mother = mother

    def __getitem__(self, i: Union[int, slice]):
        if not self._enable:
            raise RuntimeError("You can't call iloc with dataset that contains more than one exp !")
        if isinstance(i, int):
            if self.mother.tvp_dim == 0:
                return (self.mother.u[0][i], self.mother.y[0][i])
            else:
                return (self.mother.u[0][i], self.mother.y[0][i], self.mother.tvp[0][i])
        return DataSet(self.mother.u[0][i], self.mother.y[0][i], dt=self.mother.dt, tvp=None if self.mother.tvp_dim==0 else  self.mother.tvp[0][i] )

    def __len__(self):
        return len(self.u)


class DataSet(Sequence):
    @typechecked
    def __init__(self, u: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, List[np.ndarray]], dt=1.0, tvp=None, name=None):
        self.dt = dt

        # Check if u and y are consistent
        assert (isinstance(u, Iterable) and not isinstance(u, np.ndarray) ) == (isinstance(y, Iterable) and not isinstance(y, np.ndarray) ) , "u and y must be consistent in size" 

        # Check if u and s are consistent (if needed)
        if (not tvp is None):
            assert (isinstance(u, Iterable) and not isinstance(u, np.ndarray) ) == (isinstance(tvp, Iterable) and not isinstance(tvp, np.ndarray) ) , "u and tvp must be consistent in size" 

        # if multi exp check one by one
        u_dim = None
        y_dim = None
        tvp_dim = None

        self.name = name

        if isinstance(u, Iterable) and not isinstance(u, np.ndarray):

            assert len(u) == len(y), "Exp size between u and y must be consistent"
            if tvp is not None:
                assert len(tvp) == len(u), "Exp size between u, y and tvp must be consistent" 

            if tvp is None:
                tvp  = [None, ]*len(u)

            for local_u, local_y, local_tvp in zip(u,y, tvp):
                DataSet._check_couple(local_u, local_y, local_tvp)

                u_dim = local_u.shape[-1] if u_dim is None else u_dim
                y_dim = local_y.shape[-1] if y_dim is None else y_dim
                tvp_dim = 0 if local_tvp is None else (local_tvp.shape[-1] if tvp_dim is None else tvp_dim  )
                assert u_dim == local_u.shape[-1], "u shape must be consistent between experience"
                assert y_dim == local_y.shape[-1], "y shape must be consistent between experience"
                if tvp_dim > 0:
                    assert tvp_dim == local_tvp.shape[-1], "y shape must be consistent between experience"
                else:
                    assert local_tvp is None, "TVP must be None or valued at each experience"
        else:
            DataSet._check_couple(u, y, tvp)
            u_dim, y_dim, tvp_dim = u.shape[-1], y.shape[-1], 0 if tvp is None else tvp.shape[-1]
            u, y, tvp = [u,], [y,], [tvp,]

        self.u, self.y = u, y
        self.u_dim, self.y_dim, self.tvp_dim = u_dim, y_dim, tvp_dim
        self.tvp = tvp
        self.nb_exp = len(self.u)

        self.iloc = DatasetIterator(self)

    @classmethod
    def _check_couple(cls, u, y, tvp):
        assert len(u.shape) == 2, "u must be a 2D or 3D tensor" 
        assert len(y.shape) == 2, "y must be a 2D or 3D tensor" 
        if not (tvp is None):
            assert len(tvp.shape) == 2, "tvp must be a 2D or 3D tensor"

        assert y.shape[-2] == u.shape[-2], "y and u must be consistent in there time domain"
        assert (tvp is None) or tvp.shape[-2] == u.shape[-2], "tvp and u must be consistent in there time domain"

    def get_split_data(self, K, offset):
        us = []
        ys = []
        tvps = []
        for idx, exp_size in enumerate(self.get_size()):
            nb_step = exp_size-offset-K
            u_extended = [self.u[idx][i:i+offset+K] for i in range(nb_step)]
            y_extended = [self.y[idx][i:i+offset+K] for i in range(nb_step)]
            us.extend(u_extended)
            ys.extend(y_extended)
            if self.tvp_dim > 0:
                tvp_extended = [self.tvp[idx][i:i+offset+K] for i in range(nb_step)]
                tvps.extend(tvp_extended)
        return DataSet(us, ys, self.dt, tvps if self.tvp_dim > 0 else None)

    def get_dense_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Check the largest experience
        max_length = max([element.shape[0] for element in self.u])

        final_u = np.zeros((len(self.u), max_length, self.get_u_dim()))
        final_y = np.zeros((len(self.y), max_length, self.get_y_dim()))
        final_tvp = None if self.tvp_dim == 0 else np.zeros((len(self.tvp), max_length, self.get_tvp_dim()))

        final_mask = np.zeros((len(self.u), max_length))
        if self.tvp_dim == 0:
            for idx, (u, y) in enumerate(zip(self.u, self.y)):
                final_mask[idx, 0:len(u)] = 1.0
                final_u[idx, 0:len(u), :] = u
                final_y[idx, 0:len(y), :] = y
            return final_u, final_y, final_mask
        else:
            for idx, (u, y, tvp) in enumerate(zip(self.u, self.y, self.tvp)):
                final_mask[idx, 0:len(u)] = 1.0
                final_u[idx, 0:len(u), :] = u
                final_y[idx, 0:len(y), :] = y
                final_tvp[idx, 0:len(tvp), :] = tvp

            return final_u, final_y, final_tvp, final_mask

    def get_u_dim(self):
        return self.u_dim

    def get_y_dim(self):
        return self.y_dim

    def get_tvp_dim(self):
        return self.tvp_dim

    def get_size(self) -> List[int]:
        return [element.shape[0] for element in self.u]

    def __getitem__(self, i: int):
        return DataSet(self.u[i], self.y[i], dt=self.dt, tvp=self.tvp[i])

    def __len__(self):
        return len(self.u)

    def __add__(self, dataset):
        assert isinstance(dataset, DataSet), "Can only add DataSet obj together !"

        assert dataset.get_u_dim() == self.get_u_dim(), f"U dim must be compatible ! (left={ self.get_u_dim()}, right{dataset.get_u_dim() })"
        assert dataset.get_y_dim() == self.get_y_dim(), f"Y dim must be compatible ! (left={ self.get_y_dim()}, right{dataset.get_y_dim() })"
        assert dataset.get_tvp_dim() == self.get_tvp_dim(), f"TVP dim must be compatible ! (left={ self.get_tvp_dim()}, right{dataset.get_tvp_dim() })"

        assert self.dt == dataset.dt, "dt must be equals"

        return DataSet(self.u+dataset.u, self.y + dataset.y, self.dt, tvp=self.tvp + dataset.tvp)

    def iter_exp(self):
        if self.tvp_dim == 0:
            return zip(self.u, self.y)
        else:
            return zip(self.u, self.y, self.tvp)

    def __str__(self) -> str:

        name = str(id(self))[0:5] if self.name is None else self.name

        result = f"Dataset : {name}, dt={self.dt}, u_dim={self.get_u_dim()}, y_dim={self.get_y_dim()}, nb exp={len(self.u)}\n"
        result += f"Data sample (from exp1/{len(self.u)}):\n"

        table = Texttable()
        table.set_cols_align(["c"] * (1 + self.get_y_dim()+self.get_u_dim()))
        table.header(["t", ]+[f"y_{i}" for i in range(self.get_y_dim())]+[f"u_{i}" for i in range(self.get_u_dim())])
        table.set_deco(Texttable.HEADER | Texttable.VLINES | Texttable.HLINES)

        if self.u[0].shape[0] > 8:
            for idx in range(4):
                table.add_row([idx*self.dt, ] + self.y[0][idx].tolist() + self.u[0][idx].tolist())
                if idx == 1:
                    table.set_deco(Texttable.HEADER | Texttable.VLINES)
            table.add_row(["."] * (1+self.get_y_dim()+self.get_u_dim()))
            table.add_row(["."] * (1+self.get_y_dim()+self.get_u_dim()))

            for idx in range(-4, 0):
                table.add_row([(self.y[0].shape[0]-idx)*self.dt, ] + self.y[0][idx].tolist() + self.u[0][idx].tolist())

        else:
            for idx, (y, u) in enumerate(zip(self.y[0], self.u[0])):  
                table.add_row([idx*self.dt, ] + self.y[0][idx].tolist() + self.u[0][idx].tolist())
                if idx == 1:
                    table.set_deco(Texttable.HEADER | Texttable.VLINES)

        result += table.draw()
        return result

    def __repr__(self) -> str:
        return self.__str__()
