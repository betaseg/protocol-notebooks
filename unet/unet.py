import numpy as np
from six.moves import range, zip, map, reduce, filter
from six import string_types
from csbdeep.models import CARE, Config
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict, backend_channels_last
from csbdeep.data import PercentileNormalizer, PadAndCropResizer
from csbdeep.utils.tf import IS_TF_1, CARETensorBoardImage, keras_import
from csbdeep.internals import train
from csbdeep.internals.nets import custom_unet
from csbdeep.internals.predict import predict_tiled, tile_overlap, Progress, total_n_tiles

Adam = keras_import('optimizers', 'Adam')
K = keras_import("backend")

class UNetConfig(Config):
    def __init__(self, **kwargs):
        kwargs.setdefault("train_class_weight", (1,1))
        kwargs.setdefault("unet_kern_size", 3)
        kwargs.setdefault("n_channel_out", 1)
        kwargs.setdefault("unet_batch_norm", False)
        kwargs.setdefault("unet_dropout", 0.)
        
        super().__init__(allow_new_parameters = True, **kwargs)
        self.probabilistic = False
        self.unet_residual = False
        self.train_loss = "binary_crossentropy" if self.n_channel_out==1 else "categorical_crossentropy" 
        self.unet_last_activation = "sigmoid" if self.n_channel_out==1 else "softmax" 

    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.
        Returns
        -------
        bool
            Flag that indicates whether the current configuration values are valid.
        """
        def _is_int(v,low=None,high=None):
            return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
            )

        ok = {}
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except Exception as e :
            print(e)
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)
        ok['probabilistic'] = isinstance(self.probabilistic,bool)

        ok['unet_residual'] = not self.unet_residual
        ok['unet_n_depth']         = _is_int(self.unet_n_depth,1)
        ok['unet_kern_size']       = _is_int(self.unet_kern_size,1)
        ok['unet_n_first']         = _is_int(self.unet_n_first,1)
        ok['unet_last_activation'] = self.unet_last_activation in ('sigmoid', 'softmax')
        ok['unet_input_shape'] = (
                isinstance(self.unet_input_shape,(list,tuple))
            and len(self.unet_input_shape) == self.n_dim+1
            and self.unet_input_shape[-1] == self.n_channel_in
            # and all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1]))
        )
        ok['train_loss'] = self.train_loss in ('binary_crossentropy','categorical_crossentropy')
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)


        ok['train_class_weight'] = len(self.train_class_weight) == max(2,self.n_channel_out)

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())        


def weighted_bce(weights=(1,1)):
    def _loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(weights[1]*y_true*K.log(y_pred) + weights[0]*(1-y_true)*K.log(1-y_pred))
        return bce
    return _loss

def weighted_cce(weights=(1,1)):
    weights = K.variable(weights)
    def _loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return _loss


class UNet(CARE):
    @property
    def _config_class(self):
        return UNetConfig

    def _build(self):
        return custom_unet(input_shape = self.config.unet_input_shape ,
                       last_activation = self.config.unet_last_activation,
                       n_depth         = self.config.unet_n_depth,
                       n_filter_base   = self.config.unet_n_first,
                       kernel_size     = (self.config.unet_kern_size,)*self.config.n_dim,
                       pool_size       = (2,)*self.config.n_dim,
                       n_channel_out   = self.config.n_channel_out,
                       residual        = self.config.unet_residual,
                       activation      = "relu",
                       dropout         = self.config.unet_dropout,
                       batch_norm      = self.config.unet_batch_norm,
                       prob_out        = False)
    
    def prepare_for_training(self, optimizer=None, **kwargs):
        tmp = self.config.train_loss 
        self.config.train_loss = 'mse'
        optimizer = Adam(lr=self.config.train_learning_rate)
        super().prepare_for_training(optimizer=optimizer,**kwargs)
        self.config.train_loss = tmp

        loss = weighted_bce(self.config.train_class_weight) if self.config.train_loss=="binary_crossentropy" else weighted_cce(self.config.train_class_weight)

        loss = weighted_bce(self.config.train_class_weight)
        self.keras_model.compile(optimizer=optimizer, loss=loss)
    
    def train(self, X,Y, validation_data, data_gen = None,  epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.
        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.
        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.
        """
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        if data_gen is None:
            n_train, n_val = len(X), len(validation_data[0])
            frac_val = (1.0 * n_val) / (n_train + n_val)
            frac_warn = 0.05
            if frac_val < frac_warn:
                warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))

            axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
            ax = axes_dict(axes)
            for a,div_by in zip(axes,self._axes_div_by(axes)):
                n = X.shape[ax[a]]
                if n % div_by != 0:
                    raise ValueError(
                        "training images must be evenly divisible by %d along axis %s"
                        " (which has incompatible size %d)" % (div_by,a,n)
                    )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        if (self.config.train_tensorboard and self.basedir is not None and
            not IS_TF_1 and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks)):
            self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                       log_dir=str(self.logdir/'logs'/'images'),
                                                       n_images=3, prob_out=self.config.probabilistic))

        if data_gen is None:
            data_gen = train.DataWrapper(X, Y, self.config.train_batch_size, length=epochs*steps_per_epoch)
            
        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        history = fit(iter(data_gen), validation_data=validation_data,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=self.callbacks, verbose=1)
        self._training_finished()

        return history    

    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image to predict restored image.
        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image
        axes : str
            Axes of the input ``img``.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.data.Resizer` or None
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes a tuple of the number of tiles for every image axis.
            Note that if the number of tiles is too low, it is adaptively increased until
            OOM errors are avoided, albeit at the expense of runtime.
            A value of ``None`` denotes that no tiling should initially be used.
        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes semantics are the same as in the input image. Only if the output is multi-channel and
            the input image didn't have a channel axis, then output channels are appended at the end.
        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        # axes = axes_check_and_normalize(axes,img.ndim)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes,img.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C','') # axes eligible for tiling

        _permute_axes = self._make_permute_axes(img_axes_in, net_axes_in, net_axes_out, img_axes_out)
        # _permute_axes: (img_axes_in -> net_axes_in), undo: (net_axes_out -> img_axes_out)
        x = _permute_axes(img)
        # x has net_axes_in semantics
        x_tiling_axis = tuple(axes_dict(net_axes_in)[a] for a in tiling_axes) # numerical axis ids for x

        channel_in = axes_dict(net_axes_in)['C']
        channel_out = axes_dict(net_axes_out)['C']
        net_axes_in_div_by = self._axes_div_by(net_axes_in)
        net_axes_in_overlaps = self._axes_tile_overlap(net_axes_in)
        self.config.n_channel_in == x.shape[channel_in] or _raise(ValueError())

        # TODO: refactor tiling stuff to make code more readable

        def _total_n_tiles(n_tiles):
            n_block_overlaps = [int(np.ceil(1.* tile_overlap / block_size)) for tile_overlap, block_size in zip(net_axes_in_overlaps, net_axes_in_div_by)]
            return total_n_tiles(x,n_tiles=n_tiles,block_sizes=net_axes_in_div_by,n_block_overlaps=n_block_overlaps,guarantee='size')

        _permute_axes_n_tiles = self._make_permute_axes(img_axes_in, net_axes_in)
        # _permute_axes_n_tiles: (img_axes_in <-> net_axes_in) to convert n_tiles between img and net axes
        def _permute_n_tiles(n,undo=False):
            # hack: move tiling axis around in the same way as the image was permuted by creating an array
            return _permute_axes_n_tiles(np.empty(n,np.bool),undo=undo).shape

        # to support old api: set scalar n_tiles value for the largest tiling axis
        if np.isscalar(n_tiles) and int(n_tiles)==n_tiles and 1<=n_tiles:
            largest_tiling_axis = [i for i in np.argsort(x.shape) if i in x_tiling_axis][-1]
            _n_tiles = [n_tiles if i==largest_tiling_axis else 1 for i in range(x.ndim)]
            n_tiles = _permute_n_tiles(_n_tiles,undo=True)
            warnings.warn("n_tiles should be a tuple with an entry for each image axis")
            print("Changing n_tiles to %s" % str(n_tiles))

        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)

        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int,n_tiles))
        n_tiles = _permute_n_tiles(n_tiles)
        (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
            _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))
        # n_tiles_limited = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
        # if any(np.array(n_tiles) != np.array(n_tiles_limited)):
        #     print("Limiting n_tiles to %s" % str(_permute_n_tiles(n_tiles_limited,undo=True)))
        # n_tiles = n_tiles_limited
        n_tiles = list(n_tiles)


        # normalize & resize
        x = normalizer.before(x, net_axes_in)
        x = resizer.before(x, net_axes_in, net_axes_in_div_by)

        done = False
        progress = Progress(_total_n_tiles(n_tiles),1)
        c = 0
        while not done:
            try:
                # raise tf.errors.ResourceExhaustedError(None,None,None) # tmp
                x = predict_tiled(self.keras_model,x,axes_in=net_axes_in,axes_out=net_axes_out,
                                  n_tiles=n_tiles,block_sizes=net_axes_in_div_by,tile_overlaps=net_axes_in_overlaps,pbar=progress)
                # x has net_axes_out semantics
                done = True
                progress.close()
            except tf.errors.ResourceExhaustedError:
                # TODO: how to test this code?
                # n_tiles_prev = list(n_tiles) # make a copy
                tile_sizes_approx = np.array(x.shape) / np.array(n_tiles)
                t = [i for i in np.argsort(tile_sizes_approx) if i in x_tiling_axis][-1]
                n_tiles[t] *= 2
                # n_tiles = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
                # if all(np.array(n_tiles) == np.array(n_tiles_prev)):
                    # raise MemoryError("Tile limit exceeded. Memory occupied by another process (notebook)?")
                if c >= 8:
                    raise MemoryError("Giving up increasing number of tiles. Memory occupied by another process (notebook)?")
                print('Out of memory, retrying with n_tiles = %s' % str(_permute_n_tiles(n_tiles,undo=True)))
                progress.total = _total_n_tiles(n_tiles)
                c += 1

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        x.shape[channel_out] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x, net_axes_out)

        x = _permute_axes(x,undo=True)
        return x


if __name__ == '__main__':
    
    conf = UNetConfig(axes = "ZYX",
                      n_channel_in = 1,
                      train_class_weight = (1,10))

    model = UNet(conf, None,None)

    # create random single channel binary output masks
    Y = np.random.randint(0,2,(32,64,64,64,1))
    # and input 
    X = Y + .1*np.random.normal(0,1,Y.shape)

    model.train(X,Y, validation_data = [X,Y], epochs = 1, steps_per_epoch=10)
