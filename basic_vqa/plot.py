'''
plot.py: Code to implement plotting for experiments
'''
import numpy as np
import matplotlib.pyplot as plt

class Line( object ):

    def __init__( self, x, y, label=None, fmt=None ):
        self.x = x
        self.y = y
        self.label = label
        self.fmt = fmt

    def plot( self, ax ):
        if self.fmt:
            ax.plot( self.x, self.y, self.fmt, label=self.label )
        else:
            ax.plot( self.x, self.y, label=self.label )

class LineError( Line ):
    def __init__( self, x, y, err, **kwargs ):
        super().__init__( x, y, **kwargs )
        assert self.fmt
        self.err = err
        self.err_every = kwargs.get( 'err_every', None )

    def plot( self, ax ):
        ax.errorbar( self.x, self.y, label=self.label,
                    yerr=self.err, errorevery=self.err_every,
                    fmt=self.fmt )


class LinesPlot( object ):

    def __init__( self, lines, **kwargs ):
        self.lines = lines
        self.title = kwargs.get( 'title', None )
        self.x_label = kwargs.get( 'xlabel', None )
        self.y_label = kwargs.get( 'ylabel', None )
        self.legend = kwargs.get( 'legend', False )

    def plot( self, ax ):
        ax.set_title( self.title )
        for line in self.lines:
            line.plot( ax )
        ax.set( xlabel=self.x_label, ylabel=self.y_label )
        if self.legend:
            ax.legend(loc='best')

class Graph( object ):

    def __init__( self, plots, **kwargs ):
        assert isinstance( plots, list )
        self.plots = plots
        self.title = kwargs.get( 'title', 'A title' )
        self.shape = kwargs.get( 'shape', (1, ))
        self.figsize = kwargs.get( 'figsize', None )
        self.filename = kwargs.get( 'filename', 'a-plot.png' )

    def plot( self ):
        rows, cols = self.shape
        fig = plt.figure()
        fig.suptitle( self.title )
        for row in range( rows ):
            for col in range( cols ):
                i = row * cols + col
                p1 = self.plots[ i ]
                ax = fig.add_subplot( rows, cols, i+1 )
                self.plots[ i ].plot( ax )

        fig.subplots_adjust( wspace=0.4 )
        fig.savefig( self.filename )
        plt.close( fig )
        #plt.show()


def plot_loss_acc( loss, acc, prefix, filename ):
    '''
    Plot loss and accuracy vs epochs
    '''
    x = np.arange( 1, len( loss )+1, 1 )
    # create lines for loss
    loss_line = Line( x, loss, label='loss' )
    loss_dict = {
        'xlabel': 'epochs',
        'ylabel': 'loss',
        'legend': True,
    }
    lines = [ loss_line ]
    loss_plot = LinesPlot( lines, **loss_dict )
    
    # create lines for accuracy
    acc_line = Line( x, acc, label='acc' )
    acc_dict = {
        'xlabel': 'epochs',
        'ylabel': 'accuracy',
        'legend': True,
    }
    lines = [ acc_line ]
    acc_plot = LinesPlot( lines, **acc_dict )
    
    # create graph
    plots = [ loss_plot, acc_plot ]
    graph_dict = {
        'title' : f'{prefix}: Loss and Accuracy vs. epochs',
        'shape' : ( 1, 2 ),
        'filename' : filename,
    }
    graph = Graph( plots, **graph_dict )
    graph.plot()
