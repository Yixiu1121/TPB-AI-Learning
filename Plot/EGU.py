import matplotlib.pyplot as plt
import matplotlib.text as mtext


# write your code related to basemap here
plt.rcParams["font.family"] = 'Times New Roman'
# plt.show()

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title
    
plt.legend(title='Metric',label=["Line 1","Line 2"])
[line1] = plt.plot(range(10), label="Line 1")
plt.legend(title='Metric2')
[line2] = plt.plot(range(10, 0, -1), 'o', color='red', label="Line 2")


plt.show()