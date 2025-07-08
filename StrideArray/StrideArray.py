import random
import operator
import itertools


from math import prod, sqrt


chain = itertools.chain.from_iterable
accumulate = itertools.accumulate


class Array:
    _PRINT_SIGFIGS = 5
    _FULL_PRINT = True
    _PRESENTATION = 'f'
    __slots__ = ("list", "shape", "_rows", "_cols", "strides", 
                 "ndim", "parent","_transposed", "_slice", 
                 "_math_jump_table", "_matmul_jump_table")

    def __init__(self, container):
        if not isinstance(container, list):
            container = list(container)
        elems = len(container)
        self.list = container
        self.shape = (elems,)
        self._rows = 1
        self._cols = elems
        self.strides = (elems,)
        self.ndim = 1 #because we first assume a container passed is just a container
        self.parent = None
        self._slice = (0,0,1)
        self._transposed = False
        self._math_jump_table = {1:self._scalar_math , 
                                 2:self._same_shape_math, 3:self._broadcast_math}
        self._matmul_jump_table = {False :self._mat_times_mat, True:self._mat_times_vec}
        
    def __sizeof__(self):
        return sum(getattr(self, x).__sizeof__() for x in self.__slots__ if x != "parent")
    
    def __len__(self):
        return self.shape[0]
    
    def _next_chunk(self):
        for i in range(len(self)):
            yield self[i : i + 1]
            
    def _next_mat(self, reshape=False):
        rows, cols = self._rows, self._cols
        matsize = rows * cols
        num_mats = self.numel()//matsize
        if reshape:
            trans_func = lambda arr, r, c: Array(arr).reshape(r, c)
        else:
            trans_func = lambda arr, r, c : arr
        for i in range(num_mats):
            yield trans_func(self.list[i*matsize:(i+1) *matsize], rows, cols)
            
    def get_mats(self, reshape=False):
        yield from self._next_mat(reshape=reshape)
        
    def _next_row(self,reshape=False):
        row_length = self._cols 
        how_many_rows = self.numel()//row_length
        if reshape:
            trans_func = lambda arr, r: Array(arr).reshape(r)
        else:
            trans_func = lambda arr, r : arr
        for i in range(1, how_many_rows+1):
            
            yield trans_func(self.list[(i - 1) * row_length  : i * row_length], row_length)
            
    def get_rows(self, reshape=False):
        it = self._next_row(reshape) if not self._transposed else self._next_col()
        yield from it
        
    def _next_col(self,):
        csize = self._rows * self._cols
        col_stride = self.strides[-2]
        if len(self.shape) <3:
            for colnum in range(self._cols):
                yield self.list[colnum :csize:col_stride]
        else:
            for mat in self.get_mats(True):
                for colnum in range(mat._cols):
                    yield mat.list[colnum :csize:col_stride]
                    
    def get_cols(self):
        if len(self.shape) < 2:
            yield from iter(self.list)
        else:
            yield from self._next_col() if not self._transposed else self._next_row()
        
    def __iter__(self,):
        if len(self.shape) < 2:
            yield from iter(self.list)
        else:
            yield from self._next_chunk()
            
    def __repr__(self):
        if len(self.shape) == 1:
            return self.list.__repr__()
        lines = []
        line_length = self.shape[-1]
        sigfigs = Array._PRINT_SIGFIGS
        fullprint = Array._FULL_PRINT
        presentation = Array._PRESENTATION
        p_types = dict(G="General pt.", F='Fixed pt.', E="Scientific")
        if self.ndim>1:
            rows = self.shape[-2]
            cols = self.shape[-1]
            chunk_size = rows * cols
        else:
            chunk_size = self.shape[-1]
        for i,chunk in enumerate(self.get_rows()):
            if fullprint:
                lines.append(F"[{' '.join(str(x) for x in chunk)}]\n")
            else:
                lines.append(F"[{' '.join(f'{x:.{sigfigs}{presentation}}' if isinstance(x,float) else str(x) for x in chunk)}]\n")
            if not ((i+1) * line_length) % chunk_size:
                lines.append("\n")
        if fullprint:
            lines.append(F"Shape: {self.shape}")
        else:
            lines.append(F"Shape: {self.shape}\nPrint precision: 10^{-sigfigs} ({p_types[presentation.upper()]})")
        return ''.join(lines)
    
    def _get_slice(self, index, use_parent = False, jflat = False, is_int_index = False):
        shape = self.shape
        strides = self.strides if not use_parent else self.parent.strides
        index = [index,] if not hasattr(index, "__len__") else index
        len_i = len(index)
        diff = len(shape) - len_i
        flat_index = sum(ind*stride for ind, stride in zip(index, strides))
        if jflat and not diff:
            return flat_index
        if jflat and diff:
            return (flat_index, flat_index+prod(shape[len_i:]))
        if not diff:
            return flat_index, False, None, len_i
        return (flat_index, flat_index+prod(shape[len_i:])), True, shape[len_i:], len_i
    
    def _construct_arr(self, tup, shape, numel):
        start, end = tup
        if end > numel:
            raise IndexError(F"Index {end} is OOB for array with {numel} elements")
        chunk = self.list[start:end]
        ret = Array(chunk).reshape(*shape)
        ret.parent = self 
        ret._slice = (start, end, 1)
        return ret
    
    def _check_lengths(self, chunk,setval):
        elems = 0
        if hasattr(setval, "numel"):
            elems = setval.numel()
        elif hasattr(setval, "size"):
            elems = setval.size
        elif hasattr(setval, "shape"):
            elems = prod(setval.shape)
        else:
            elems = len(setval)
        return chunk.numel() == elems
    
    def _pad_item(self, to, item):
        return [item for _ in range(to)]
    
    def __setitem__(self, key, value):
        raise NotImplementedError("Please don't do this :(")
        
    def _get_inds_from_slice(self, slicer):
        unpack_slice = lambda x : (x.start or 0, x.stop or len(self), x.step or 1)
        slice_start, slice_stop, slice_step = unpack_slice(slicer)
        our_step = self.strides[0] * slice_step
        q, r = divmod(slice_stop-slice_start, slice_step)
        indices = [slice_start + i * our_step for i in range(q + r )]
        return indices,q ,(slice_start, slice_stop, slice_step)
    
    def _convert_arr_slice_to_list_slice(self, slicer):
        unpack_slice = lambda x : (x.start or 0, x.stop or len(self), x.step or 1)
        slice_start, slice_stop, slice_step = unpack_slice(slicer)
        dim0 = (1+slice_stop - slice_start,) if slice_start else (slice_stop - slice_start,)
        fstart = self._get_slice(slice_start, jflat=True, is_int_index = True)
        fend = self._get_slice(slice_stop, jflat = True, is_int_index = True)            
        if isinstance(fstart, tuple):
            fstart = fstart[0]
        if isinstance(fend, tuple):
            fend = fend[0] if dim0[0] < self.shape[0] else fend[-1]
        if dim0[0] > 2:
            newshape = dim0 + self.shape[1:]
        else:
            newshape = self.shape[1:]
        if not slice_start:
            newshape = (slice_stop, ) + self.shape[1:]
        return slice(fstart, fend, slice_step), newshape 
    
    def _mask(self, indices):
        assert len(indices) == self.shape[0], "all indices have to be masked, babe :D"
        
    def index_info(self, index):
        is_int = False
        is_tup = False
        is_slice = False
        index_type = type(index) 
        int_info = None
        neg_index_in_tuple = False
        slice_in_tuple = False
        tup_index_newshape = None
        if index_type == int: #SLIGHTLY faster than isintance(index, type_of_thing)
            int_info = self._get_slice(index, is_int_index = True)
            return True, is_tup, is_slice, int_info, neg_index_in_tuple, slice_in_tuple,tup_index_newshape
        elif index_type == tuple:
            slice_in_tuple = any(type(x) == slice for x in index)
            if not slice_in_tuple:
                neg_index_in_tuple = all(x<0 for x in index)
            return is_int, True, is_slice,int_info, neg_index_in_tuple, slice_in_tuple,tup_index_newshape
        elif index_type == slice:
            return is_int, is_tup, True,int_info, neg_index_in_tuple, slice_in_tuple,tup_index_newshape
        return is_int, is_tup, is_slice,int_info, neg_index_in_tuple, slice_in_tuple,tup_index_newshape
    
    def __getitem__(self, index):
        """
        Currently slices are not implemented
        """
        (is_int, is_tup, is_slice,
        int_info, neg_index_in_tuple, slice_in_tuple, tup_index_newshape)  = self.index_info(index)
        numel = self.numel()
        if is_int:
            findex, int_index_is_tup, newshape, len_i = int_info
            if int_index_is_tup:
                return self._construct_arr(findex, newshape, numel)
            if findex > numel:
                raise IndexError(F"Index {index} is out of bounds")
            return self.list[findex]
        elif is_tup:
            if slice_in_tuple:
                raise IndexError("Please, just don't use slices... I beg you")
            if neg_index_in_tuple:
                raise IndexError("Please don't give negative indices, they make me cry")
            else:
                findex, int_index_is_tup, newshape, len_i = self._get_slice(index, is_int_index = True)
                if int_index_is_tup:
                    return self._construct_arr(findex, newshape, numel)
                return self.list[findex]
            
    def _broadcastable(self, other):
        if self.shape == other.shape:
            return 2
        return 3
    
    def _check_type(self, other):
        if isinstance(other, (int, float)):
            return 1
        if not hasattr(other, "shape"):
            raise TypeError("Needs to be a scalar or something with a 'shape' attribute")
        return self._broadcastable(other)
        
    def _scalar_math(self, other, func):
        ret = Array.zeros(*self.shape)
        for i, val in enumerate(self.list):
            ret.list[i] = func(val, other)
        return ret
    
    def _flat_iter(self):
        fit = chain(self.get_rows())
        for item in fit:
            yield item
            
    def _flat(self):
        yield from self._flat_iter()
        
    def _same_shape_math(self, other, func, ):
        ret = Array.zeros(*self.shape)
        for i, (val, oval) in enumerate(zip(self._flat(), other._flat())):
            ret.list[i] = func(val, oval)
        return ret
    
    def _broadcast_math(self, other, func):
        pass
    
    def __add__(self, other):
        func = operator.add
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __sub__(self, other):
        func = operator.sub
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __mul__(self, other):
        func = operator.mul
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __truediv__(self, other):
        func = operator.truediv
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __floordiv__(self, other):
        func = operator.floordiv
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __pow__(self, exp):
        if not isinstance(exp, (int, float)):
            return #raise error
        func = operator.pow
        return self._math_jump_table[self._check_type(exp)](exp, func)
    
    def __neg__(self):
        return Array([-x for x in self.list]).reshape(*self.shape)
    
    def _dot(self, v1, v2, ):
        return sum(x*y for x, y in zip(v1, v2))
    
    def _mat_times_vec(self, other, ret):
        for i, row in enumerate(self.get_rows()):
            ret.list[i] = sum(x * y for x, y  in zip(row, other.get_cols()))
        return ret
    
    def _mat_times_mat(self, other, ret):
        sm2, sm1 = ret.strides[-2], ret.strides[-1]
        retlist = ret.list
        for i, row in enumerate(self.get_rows()):
            for j, col in enumerate(other.get_cols()):
                retlist[i * sm2 + j * sm1 ] += sum(x * y for x,y in zip(row, col))
        ret.list = retlist
        return ret
    
    def _check_tensor_dims(self, other):
        self_leading_dims = self.shape[:-2] 
        if other._rows == 1 and other._cols == self._cols:
            return self_leading_dims + (self._rows,), False
        if self._cols != other._rows:
            return None
        other_leading_dims = other.shape[:-2]
        if len(other_leading_dims) > len(self_leading_dims):
            return None
        padded_other_dim = tuple(None for _ in range(len(self_leading_dims)-len(other_leading_dims))) + other_leading_dims
        ret_mat_shape = (self._rows, other._cols)
        for sdim, odim in zip(self_leading_dims, padded_other_dim):
            if odim is None or odim == sdim:
                continue
            if odim != sdim and odim !=1:
                return None
        return self_leading_dims + ret_mat_shape, self_leading_dims == other_leading_dims
        
    def _tensor_mult(self, other, v2_is_vec=False ):
        ret_shape, sdims = self._check_tensor_dims(other)
        if ret_shape is None:
            return 'faield'#raise somethingg
        oiter = other.get_mats(True)
        if v2_is_vec:
            oiter = [next(other.get_rows(True))]
        ret = []
        retextend = ret.extend
        if sdims:
            for mat, o in zip(self.get_mats(True), other.get_mats(True)):
                retextend((mat @ o).list)
            return Array(ret).reshape(*ret_shape)
        for mat in self.get_mats(True):
            for o in oiter:
                retextend((mat @ o).list)
            if not v2_is_vec:
                oiter = other.get_mats(True)
        return Array(ret).reshape(*ret_shape)
    
    def __matmul__(self, other):
        v2_is_vec = len(other.shape) < 2
        if len(self.shape)>2:
            return self._tensor_mult(other, v2_is_vec)
        if self._cols != other._rows:
            pass #raisew something lol
        ret_shape = (self._rows,) if v2_is_vec else (self._rows, other._cols)
        ret = Array.zeros(*ret_shape)
        ret = self._matmul_jump_table[v2_is_vec](other, ret)
        return ret
    
    def __eq__(self, other):
        func = operator.eq
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __lt__(self, other):
        func = operator.lt
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __le__(self, other):
        func = operator.le
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __gt__(self, other):
        func = operator.gt
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __ge__(self, other):
        func = operator.ge
        return self._math_jump_table[self._check_type(other)](other, func)
    
    def __bool__(self):
        return Array.any(self.list)
    
    def __contains__(self, item):
        if not isinstance(item, (int, float, complex)):
            return False #actually raise something lol
        return any(x==item for x in self.list)

    def __round__(self, ndigits=4):
        _round = round
        return Array([_round(x, ndigits) for x in self.list]).reshape(*self.shape)
    
    def _setup_strides(self):
        shape = self.shape[1:] 
        p = 1
        shape = shape + (1,)
        strides = []
        for i, s in enumerate(reversed(shape)):
            if i:
                p *= s
            strides.append(p)
        self.strides = list(reversed(strides))
            
    def _set_row(self, i, val):
        stride = self.strides[0]
        self.list[stride * i : (i + 1) * stride] = val

    def _get_row(self, i):
        stride = self.strides[0]
        return self.list[stride * i : (i + 1) * stride]
    
    def _set_col(self, i, val):
        csize = self._rows * self._cols
        col_stride = self.strides[-2]
        self.list[i :csize :col_stride] = val
        
    def _get_col(self, i):
        csize = self._rows * self._cols
        col_stride = self.strides[-2]
        return self.list[i :csize :col_stride]
    
    def numel(self):
        return prod(self.shape)
    
    def max(self):
        return max(self.list)
    
    def min(self):
        return min(self.list)
    
    def absmin(self):
        return min(abs(x) for x in self.list)
    
    def absmax(self):
        return max(abs(x) for x in self.list)
    
    def abs(self):
        return Array([abs(x) for x in self.list]).reshape(*self.shape)
    
    def norm(self):
        return sqrt(sum( x*x for x in self.list ))
    
    def reshape(self, *shape):
        size = shape.__len__()
        s_gt_1 = size.__gt__(1)
        if prod(shape) != self.numel(): 
            return None
        self.shape = shape
        if s_gt_1:
            self._rows = shape[-2]
            self._cols = shape[-1]
        else:
            self._cols = shape[-1]
        self._setup_strides()
        self.ndim = size
        return self

    def diag(self,ret_array = True):
        row_stride = self.strides[0]
        n = self.shape[0]
        sl = self.list
        diagvals = [sl[i * row_stride+ i] for i in range(n)]
        if not ret_array:
            return diagvals 
        ret = Array(diagvals).reshape(n)
        return ret
    
    def diagprod(self):
        return prod(self.diag(False))
    
    def diagflat(self):
        row_stride = self.strides[0]
        n = self.shape[0]
        eye = Array.eye(n)
        diag = self.diag(ret_array = False)
        for i in range(n):
            eye.list[i * row_stride + i] = diag[i] 
        return eye

    def transpose(self):
        ret = []
        rextend = ret.extend
        for col in self.get_cols():
            rextend(col)
        revshape = tuple(reversed(self.shape))
        return Array(ret).reshape(*revshape)
    
    def astype(self, newtype):
        return Array([newtype(x) for x in self.list]).reshape(*self.shape)
    
    def int(self):
        return self.astype(int)
    
    def float(self):
        return self.astype(float)
    
    def complex(self):
        return self.astype(complex)
    
    def sum(self, dim=-1):
        return sum(self.list)
    
    def index_as_flat(self, flat_index):
        return self._flat_to_fancy(flat_index)
    
    def _flat_to_fancy(self, flat_index):
        indices = []
        for dim in reversed(self.shape):
            indices.append(flat_index % dim)
            flat_index //= dim
        return tuple(reversed(indices))
    
    def where_nonzero(self, ):
        nonzero_indices = []
        append = nonzero_indices.append
        for i, item in enumerate(self.list):
            if item:
                append(self._flat_to_fancy(i))
        return nonzero_indices

    def _little_bool_comp(self, other, func):
        indices = []
        append = indices.append
        for i, (s, o) in enumerate(zip(self.list, other.list)):
            if func(s, o):
                append(self._flat_to_fancy(i))
        return indices 
    
    def where_compare(self, other, gt=False, ge=False, lt=False, le=False, custom_comp= None):
        indices = []
        if gt:
            func = operator.gt
            return self._little_bool_comp(other, func)
        elif ge:
            func = operator.ge
            return self._little_bool_comp(other, func)
        elif lt:
            func = operator.lt
            return self._little_bool_comp(other, func)
        elif le:
            func = operator.le
            return self._little_bool_comp(other, func)
        elif custom_comp is not None:
            return self._little_bool_comp(other, custom_comp)
        return indices 
    
    @property
    def T(self):
        shape = self.shape
        if len(shape)<2:
            return self
        revshape = tuple(reversed(shape))
        ret = Array(self.list.copy()).reshape(*revshape)
        ret._transposed = not self._transposed
        return ret
        
    @classmethod
    def sym_rand_2d(cls, *shape, prec = 5):
        if len(shape)!= 2:
            print(shape, len(shape))
            return #raise something
        ret = Array.zeros(*shape)
        rlist = ret.list
        rows, cols = shape
        rs = ret.strides[0]
        rand = random.random
        _round = round
        for i in range(rows):
            for j in range(i, cols):
                _rand = _round(rand(), prec)
                rlist[i * rs + j] = _rand
                rlist[j*rs + i] = _rand
        ret.list = rlist
        return ret
                
    @classmethod
    def all(cls, mat):
        return all(mat.list)
    
    @classmethod
    def all_eq(cls, mat, other):
        res = mat==other
        return all(res.list)
        
    @classmethod
    def any(cls, mat, ):
        return any(mat.list)
    
    @classmethod
    def any_eq(cls, mat, other):
        res = mat==other
        return any(res.list)
    
    @classmethod
    def all_close(cls, a, b, tol=1e-9):
        return all(abs(x-y)<tol for x,y in zip(a.list, b.list))
    
    @classmethod
    def close_to_zero(cls, a, tol=1e-9):
        return all(abs(x)<tol for x in a.list)
    
    @classmethod
    def close_to_eye(cls, a, tol=1e-9):
        eye = Array.eye(a.shape[0])
        return all(abs(x-y)<tol for x,y in zip(a.list, eye.list))
    
    @classmethod
    def _fill(cls, shape, fill_val, isfunc=False):
        if not isfunc:
            arr = cls(list(fill_val for _ in range(prod(shape))))
        else:
            arr = cls(list((fill_val() for _ in range(prod(shape)))))
        arr = arr.reshape(*shape)
        return arr
    
    @classmethod
    def zeros(cls, *shape):
        arr = cls._fill(shape, fill_val=0)
        return arr
    
    @classmethod
    def ones(cls, *shape):
        arr = cls._fill(shape, fill_val=1)
        return arr
    
    @classmethod
    def rand(cls, *shape):
        arr = cls._fill(shape, fill_val=random.random, isfunc = True)
        return arr
    
    @classmethod
    def round_rand(cls, *shape, prec=5):
        rrand = lambda : round(random.random(), prec)
        arr = cls._fill(shape, fill_val=rrand, isfunc = True)
        return arr
    
    @classmethod
    def linspace(cls, start, stop, step):
        abs_diff = abs(stop-start)
        if stop < start:
            start,stop = stop, start
        if step < 1 or step < abs_diff:
            steps = abs_diff // step
        elif step > abs_diff:
            steps = step
        steps = int(steps)
        step = abs_diff/(steps-1)
        _space  = cls(list((start + step *  interval for interval in range(steps))))
        _space = _space.reshape(len(_space.list))
        return _space
        
    @classmethod
    def arange(cls, start, stop=0, step = 1):
        if not isinstance(step, int):
            return cls.linspace(start, stop, step)
        if stop:
            _range = cls(list((range(start, stop, step))))
        if not stop and step==1:
            _range = cls(list((range(start))))
        elif not stop and step !=1:
            _range  = cls(list((range(0, start, step))))
        _range = _range.reshape(len(_range.list))
        return _range
    
    @classmethod
    def eye(cls, size):
        ret = cls.zeros(size, size)
        for i in range(size):
            ret[i,i] = 1
        return ret
    
    @classmethod
    def set_print_prec(cls, prec = 5, presentation = "f"):
        cls._FULL_PRINT = False
        cls._PRINT_SIGFIGS = prec
        cls._PRESENTATION = presentation 
        
    @classmethod
    def no_print_prec(cls, ):
        cls._FULL_PRINT=True
































