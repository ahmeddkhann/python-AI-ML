# This is a guide to learn matplotlib library through practice
# and coding exercise with detailed explanations for visualization

import matplotlib.pyplot as plt
import numpy as np 


######################################### PLOTS ################################################


# Basic Line Plot for continous data
# we will be creating two numpy arrays for x and y axis
x_values = np.array([1,2,3,4,5])
y_values = np.array([2,4,6,8,10])
# this detrmine the size of the window that will display the plot
# first parameter is for the width and second one is for height
# the size is calculated in inches. i.e 6 inches width and 3 inches height
plt.figure(figsize=(6,3))
# this line plot the values on x and y axis.
# the first paramater will print the value on the x axis.
# the second parameter will print the value on the y axis. 
plt.plot(x_values, y_values)
# this line gives title to the figure.
plt.title("Continious Data XY-Plan")
# adding label on the x-axis.
plt.xlabel("X-Axis")
# adding label on y-axis.
plt.ylabel("Y-Axis")
# this will display the figure.
# plt.show()


# Line plot with multiple lines
# Now we will be plotting multiple lines on one figure
x_data = np.array([1,2,3,4,5])
y_data1 = np.array([1,6,9,10,14])
y_data2 = np.array([2,9,12,13,5])
plt.figure(figsize=(8,4))
# this will plot y_data1 corresponding to x-axis
plt.plot(x_data, y_data1, label="Line 1")
# this will plot y_data2 corresponding to x-axis
plt.plot(x_data, y_data2, label="Line 2")
plt.title("Mapping two different dataset with 1 dataset")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# legend() creates a box type having labels of both the lines
plt.legend()
# plt.show()


# Colours and Styles
# we will be plotting lines with different colours and lines
plt.figure(figsize=(8, 4))
# 1. linestyle: It represents that how a line will be drawn.
# Different line styles: '-', '--', '-.', ':'
# 2. marker: It represents how different points will be marked.
# Different markers: 'o', 's', '^', '*', '+', 'x'
# 3. color: Gives color to the line.
# 4. markerfacecolor: Gives color to the marker.
# 5. markeredgecolor: Gives color to the marker edges.
# 6. linewidth: Gives width to the line.
# 7. markeredgewidth: Gives width to the edge
# 8: fontsize: Gives size to the font.
# 9: fontweight: Gives weight to the font such as bold or light.
# 10:fontfamily: Gives different font styles/family.
# 11:loc: Gives location to the legend on where to be displayed on the graph.

plt.plot(x_data, y_data1, 
         label="Line 1",
         color="blue",
         linestyle=":",
         linewidth=2,
         marker="o",
         markerfacecolor="pink",
         markeredgecolor="black",
         markeredgewidth=2
         )
plt.plot(x_data, y_data2, 
         label="Line 2",
         color="red",
         linestyle="--",
         linewidth=2,
         marker="^",
         markerfacecolor="purple",
         markeredgecolor="black",
         markeredgewidth=2
         )
plt.title("Plot with Line Styles and Markers",
          fontsize=14,
          fontweight="bold",
          fontfamily="serif",
          color="green"
          )
plt.xlabel("X-Axis",
           fontsize=12,
           fontweight="light",
           fontfamily="serif",
           color="blue"
           )
plt.ylabel("Y-Axis",
           fontsize=12,
           fontweight="light",
           fontfamily="serif",
           color="red"
           )
plt.legend(
    fontsize=10,
    loc="upper left"
)

# plt.show()


######################################### Scatters ################################################

# Scatter plots are used to show a relationship between two numeric values.
# instead of connecting lines, it shows the exact points.
x_scatter = np.array([1,2,3,4,5,6,7,8])
y_scatter = np.array([10,15,9,12,5,6,10,8])
plt.figure(figsize=(8,4))
plt.scatter(x_scatter, y_scatter)
plt.title("Scatter Plot")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# plt.show()

# Basic Customization of the Scatters
plt.figure(figsize=(8,4))
# s is used to adjust the size of the markers
plt.scatter(x_scatter, y_scatter,s=50, color="red")
plt.title("Basic Customized Scatter")
# plt.show()


# Advance Customization of the Scatters
# size array is just some numbers that will be used as size of the markers
# colors array are only numbers that will be used as colors.
# Paramters of Scatter:
# 1. s: size of the marker.
# 2. c: instead of color, we have to write c to pass color numbers
# 3. cmap: it is used to map the color numbers with the colors.
# viridis, viridis_r and cividis are the options that can be used
# 4. aplha: Shows how much transparent the marker should be. the value
# ranges between 0 and 1.
sizes = np.array([50, 100, 150, 200, 250, 300, 350, 400])
colors = np.array([1,2,3,4,5,6,7,8])
plt.figure(figsize=(8,4))
scatter = plt.scatter(x_scatter,y_scatter,
            s=sizes,
            c=colors,
            cmap="viridis",
            alpha=0.6
            )
plt.colorbar(scatter)
plt.title("Advanced Customized Scatter",
          color="purple",
          fontfamily="fantasy",
          fontweight="heavy"
          )
plt.xlabel("X-Axis",
           color="green",
           fontweight="normal",
           fontfamily="cursive"
           )
plt.ylabel("Y-Axis",
           color="black",
           fontweight="ultralight",
           fontfamily="monospace"
           )

# minorticks_on function:
# be default, only the large numbers or values are showed as scale on x and y
# axis. it neglects the smaller numbers. the minorticks_on function drew
# small lines between them which are useful to values between two large scal numbers.
plt.minorticks_on()
# grid is used to draw grid lines on the diagram either scatter, plot or whatever.
# parameters of the grid:
# 1. visible: determines if the grid will be displayed or not.
# 2. which: determines if the grid lines should be for only big values 
# or small values or both. accepts major, minot, both.
# 3. axis: determines on which axis the lines should be drawn. accepts
# either x , y or both.
plt.grid(
    visible=True,          
    which='minor',         
    axis='both',          
    color='gray',         
    linestyle='--',        
    linewidth=0.5,        
    alpha=0.7              
)
# plt.show()


######################################### Histograms ################################################

# Histograms distributes data by dividing it into bins and each bin represents frequency.
# It grou values into ranges called bins and counts how many values falls into that range.
# Basic Histograms
histogram_data = np.array([10, 20, 30, 40, 25, 50, 40, 80, 75, 90,
                           80, 100, 60, 73, 88, 54, 48, 65, 90, 35])

plt.figure(figsize=(8,4))
plt.hist(histogram_data, bins=10)
plt.title("Histogram")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# plt.show()


# Advanced Histograms
histogram_data1 = np.array([3,23,18,30,29,28,27,39,20,10,
                            4,15,23,20,35,44,37,28,36,48])
plt.figure(figsize=(8,4))
# Paramteres Explaination:
# 1. Bins: It determines how many number of bins should be there.
# 2. Edge color determines the color of the edges of the histograms
# so that one histogram can be distinct from the another one.
# 3. rwidth: It determines the width of the histograms. by defualt,
# it is 1. and it ranges between 0 and 1, decreasing the value creates
# some sort of distance between two histograms which is essential to
# make disticntion among the histograms.
plt.hist(histogram_data1, 
         bins=10, 
         alpha=0.5,
         label="Data1", 
         color="red", 
         edgecolor="black",
         rwidth=0.9)
plt.title("Multiple Histograms")
# Instead of creating matplotlib own ranges,  we can give
# custom ranges as well.
# 1. xticks: it determines ranges on the x axis.
# 2. yticks: it determines ranges on y axis.
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.yticks([0,1,2,3,4])
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# plt.show()




######################################### Barcharts ################################################
############################# Horizantal and Verical Barcharts #####################################

# Barcharts are used to compare different categories through Bars
# Each bar represents seperate category
# There are two types of barcharts i.e Vertical and Horizantal
# Vertical Barcharts
barchart_categories = np.array(["A", "B", "C", "D", "E", "F"])
barchart_values = np.array([13, 25, 20, 27, 17, 22])
plt.figure(figsize=(8,4))
# the bar amd barh function needs two parameters. i.e categories and values
# the first parameter is always categories while the second is value. 
plt.bar(barchart_categories, barchart_values)
plt.title("Vertical Barcharts")
plt.xlabel("Categories")
plt.ylabel("Values")
# plt.show()
# Horizantal Barcharts
plt.figure(figsize=(8,4))
plt.barh(barchart_categories, barchart_values)
plt.title("Horizantal Barcharts")
plt.xlabel("Categories")
plt.ylabel("Values")
# plt.show()

# Advance barcharts with some customization
# we will be comparing numbers of students in each section of a class
sections = np.array(["A", "B", "C", "D", "E", "F", "G", "H"])
school1 = np.array([37, 42, 29, 36, 48, 45, 40, 35])
school1_bar_colors = np.array(["orange", "green", "yellow", "pink", "purple", "blue", "red", "brown"])
# Hatches are used to create some sort of design inside the bar. It is usely used
# for the color blind people or if the print is in black and white
# so the distinction can be done on the basis of design in it rather than colors.
hatches = ["//", "\\\\", "||", "--", "++", "xx", "oo", ".."]
# Vertical Barchart
plt.figure(figsize=(10,5))
plt.bar(sections, school1, color=school1_bar_colors,
                           alpha=0.7, 
                           edgecolor="black", 
                           width=0.6,
                           linewidth=2,
                           linestyle="--",
                           hatch=hatches
                           )
plt.title("Vertical Advance Barcharts")
plt.xlabel("Sections")
plt.ylabel("No.of Students in each section")
plt.minorticks_on()
# plt.show()
# Horizantal Barchart
plt.figure(figsize=(10,5))
plt.barh(sections, school1, color=school1_bar_colors,
                           alpha=0.7, 
                           edgecolor="black", 
                           linewidth=2,
                           linestyle="--",
                           hatch=hatches
                           )
plt.title("Vertical Advance Barcharts")
plt.xlabel("Sections")
plt.ylabel("No.of Students in each section")
plt.minorticks_on()
# plt.show()



######################################### Pie Charts ################################################


# pie charts shows parts of whole. each section represents a proportion.
# Basic Pie Chart
pie_sizes = np.array([10,15,12,23,20])
pie_labels = np.array(["A", "B", "C", "D", "E"])
plt.figure(figsize=(8,4))
plt.pie(pie_sizes, labels=pie_labels)
plt.title("Pie Chart")
# plt.show()
# Pie Chart with advance customization
# we will check the propotion of each grade that students scored
no_of_students_of_each_grade = np.array([28,35,40,33,42,34,15])
students_grades = np.array(["A+", "A", "B", "C", "D", "E", "F"])
# explode is used to seperate slices from the centre.
# we are seperating A+ and F grade slices which are at first 
# and last so those two values are not zero.
explode = np.array([0.1,0,0,0,0,0,0.1])
colors = np.array([ "gold","skyblue", "lightgreen", "orange","lightcoral","plum", "red"])
plt.figure(figsize=(8,4))
# Paramteres Explaination:
# 1. labels: add labels to each slice like A+, A etc.
# 2. autopct: adds the percentage to each slice. the current
# one will print one decimal percentage like 12.1%.
# 3. startangle: it determines from which angle the chart will starts.
# 4. colors: adds custom colors to each slice.
# 5. shadow: it adds shadow to the pie chart.
# 6. pctdistance: determines that on how much distance the percentage
# will be written from the centre.
# 7: labeldistance:determines that on how much distance the labels
# will be written from the slice.
# 8. explode: it seperates the slides from the centre.
# 9. wedgeprops: accepts dict that adds properties to edges.
# 10: textprops: accepts dict that adds properties to text.
plt.pie(no_of_students_of_each_grade, 
                labels=students_grades,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                shadow=False,
                pctdistance=0.75,
                labeldistance=1.05,
                explode=explode,
                wedgeprops={
                    "edgecolor":"black",
                    "linewidth": 1.5
                },
                textprops={
                    "fontsize":12,
                    "fontweight": "bold"
                }
                )
plt.title("Students Grades")
# bbox_to_anchor: sometimes the legend overlayes over the chart
# it is used to move the legend away from the chart.
plt.legend(students_grades,
           title="Grades",
           loc="upper left", 
           bbox_to_anchor=(1, 0.5))
plt.title("Advanced Customized Pie Chart",
          color="purple",
          fontfamily="fantasy",
          fontweight="bold",
          )
# plt.show()



######################################### Box Plots ################################################

# box plots shows distribution of data and outliers in it
# shows min, Q1, median, Q3, max
data_box1 = np.array([10, 20, 30, 40, 50])
data_box2 = np.array([15, 25, 35, 45, 55])
data_box3 = np.array([5, 15, 25, 35, 100]) 
plt.figure(figsize=(8,4))
plt.boxplot([data_box1, data_box2, data_box3], 
            label=["Data 1", "Data 2", "Data 3"])
plt.ylabel("Simple Box Plot")
# plt.show()
# Advance Box Plot
# we will use the same arrays but will display an advance
# box plots
plt.figure(figsize=(8,4))
# Paramters Explaination:
# 1. showmeans: This determines if the mean value should be displayed or not
# 2. meanprops: determines the properties of the mean.
# 3. medianprops: determines the properties of the median.
# 4. whiskerprops: determines the properties of the whisker.
# 5. capprops: determines the properties of the cap at edge of whiskers.
# 6. flierprops: flier are theoutliers in data. it determines the
# properties of fliers/outliers in data to be displayed.
plt.boxplot([data_box1, data_box2, data_box3], 
            label=["Data 1", "Data 2", "Data 3"],
                # Show mean
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 8
            },
            medianprops={
                "color": "red",
                "linewidth": 2
            },
            whiskerprops={
                "color": "black",
                "linewidth": 1.5
            },
            capprops={
                "color": "black",
                "linewidth": 1.5
            },
            flierprops={
                "marker": "o",
                "markerfacecolor": "orange",
                "markeredgecolor": "black",
                "markersize": 8
            }
            )
plt.ylabel("Advance Box Plot with Customization")
# plt.show()



######################################### Violin Plots ################################################

# Similar to box plot but shows full distribution
# box plots and violin plots are almost similar but
# violin plots are more detailed while box plots are
# summarized ones.
plt.figure(figsize=(8, 4))
# Paramaters Explaination:
# 1. positions: determine the positions of the violent
# 2. showmeans: determines the visibility of the mean.
# 3. showmedians: determines the visibility of the median.
# 4. showestrema: determines the visibility of the outliers.
plt.violinplot([data_box1, data_box2, data_box3],
               positions=[1, 2, 3], 
               showmeans=True, 
               showmedians=True,
               showextrema=True
               )
plt.title("Violin Plot")
plt.ylabel("Values")
plt.xticks([1, 2, 3], ['Data 1', 'Data 2', 'Data 3'])
# plt.show()


######################################### Heat Maps ################################################

# heat maps are 2D visual representations of data
# where individual values in matrix are represented as colors.
# we do not have a heatmap() function in matplotlib.
# Basic Heat Map
heatmap_data = np.array([
    [10,23,18,40],
    [34,19,42,26],
    [37,25,35,11],
    [27,29,45,17]
])
plt.figure(figsize=(8,4))
plt.imshow(heatmap_data, cmap="hot", aspect="auto")
plt.colorbar()
plt.title("Basic Heatmap")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# plt.show()
# Advance HeatMaps
plt.figure(figsize=(8,4))
# Paramters Explaination:
# 1. cmap: determines the type of colors that will be used.
# accepted options are cool, auto, coolwarm etc.
# 2. aspect:  aspect=auto is used to fit the squares inside the
# given figure rather than using own sizes of square.
# 3. vmin: determines the smallest value.
# 4. vmax: determines the largest value.
# 5. interpolation: it is used to not blend the values over the
# neighbouring values. other option is billineail.
# 6. alpha: used for transparency. can be range from 0 to 1.
# 7. origin: by default, the matplotlib shows the data from the top to
# bottom but from this parametre, we can show data from bottom to top. 
plt.imshow(heatmap_data,
           cmap="cool", 
           aspect="auto",
           vmin=0,
           vmax=50,
           interpolation="nearest",
           alpha=0.9,
           origin="lower"
           )
# colorbar is used to display a colorbar along with the values.
# label: add the values
# shrink; used to adjust the size of the heatmap.
plt.colorbar(
    label="Values",
    shrink=0.8
)
plt.title("Advance Heatmap")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# plt.show()



######################################### Sub Plots ################################################

# subplots are used to print multiple graphs or compare 
# data of multiple graphs is one window.
# below we will be printing some subplots.

subplotsdata_w = np.array([2,5,8,6,10])
subplotsdata_x = np.array([1,6,9,10,14])
subplotsdata_y = np.array([2,6,9,7,12])
subplotsdata_z = np.array([2,9,12,13,5])
y_axis_scatter = np.array([10,15,9,12,5])

# Line Plots
# Printing 2 subplots
plt.figure(figsize=(8,4))
# the most important line to understand how the plots works.
# this line tells the matplotlib, that creates 1 row, 2 columns
# and i am currently working on the 1st part. the later one will
# be (1,2,2) meaning 1 row, 2 columns and 2nd part.
plt.subplot(1,2,1)
plt.plot(subplotsdata_x)
plt.title("Line 1")
plt.xlabel("Line 1 X-Axis")
plt.ylabel("Line 1 Y-Axis")
plt.subplot(1,2,2)
plt.plot(subplotsdata_y)
plt.title("Line 2")
plt.xlabel("Line 2 X-Axis")
plt.ylabel("Line 2 Y-Axis")
# plt.show()

# Scatter Plots
# printing 4 subplots
plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.scatter(subplotsdata_w, y_axis_scatter)
plt.title("Scatter 1")
plt.xlabel("Scatter 1 X-Axis")
plt.ylabel("Scatter Y-Axis")
plt.subplot(2,2,2)
plt.scatter(subplotsdata_x, y_axis_scatter)
plt.title("Scatter 2")
plt.xlabel("Scatter 2 X-Axis")
plt.ylabel("Scatter 2-Axis")
plt.subplot(2,2,3)
plt.scatter(subplotsdata_y, y_axis_scatter)
plt.title("Scatter 3")
plt.xlabel("Scatter 3 X-Axis")
plt.ylabel("Scatter 3-Axis")
plt.subplot(2,2,4)
plt.scatter(subplotsdata_z, y_axis_scatter)
plt.title("Scatter 4")
plt.xlabel("Scatter 4 X-Axis")
plt.ylabel("Scatter 4-Axis")
# this line ensures that content of one plot does not
# overlay content of the another plot.
plt.tight_layout()
# plt.show()