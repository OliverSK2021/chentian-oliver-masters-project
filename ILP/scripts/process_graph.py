import graph_tool.all as gt
import sys
import os
import random

random.seed(43)

background = open("ilp.b", 'w')
positive = open("ilp.f", 'w')
negative = open("ilp.n", 'w')
test_positive = open("test.f", 'w')
test_negative = open("test.n", 'w')

b_lines = []
p_lines = []
n_lines = []
tp_lines = []
tn_lines = []

graphs_directory = sys.argv[1]

train_positive_directories = [(graphs_directory + "/" + i) for i in os.listdir(graphs_directory) if (i.startswith("train_positive") or i.startswith("valid_positive"))]
train_negative_directories = [(graphs_directory + "/" + i) for i in os.listdir(graphs_directory) if (i.startswith("train_negative") or i.startswith("valid_negative"))]

test_positive_directories = [(graphs_directory + "/" + i) for i in os.listdir(graphs_directory) if i.startswith("test_positive")]
test_negative_directories = [(graphs_directory + "/" + i) for i in os.listdir(graphs_directory) if i.startswith("test_negative")]

train_positive_files = [(j + "/" + i, True, True) for j in train_positive_directories for i in os.listdir(j) if os.path.splitext(i)[1] == ".graphml"]
train_negative_files = [(j + "/" + i, False, True) for j in train_negative_directories for i in os.listdir(j) if os.path.splitext(i)[1] == ".graphml"]
test_positive_files = [(j + "/" + i, True, False) for j in test_positive_directories for i in os.listdir(j) if os.path.splitext(i)[1] == ".graphml"]
test_negative_files = [(j + "/" + i, False, False) for j in test_negative_directories for i in os.listdir(j) if os.path.splitext(i)[1] == ".graphml"]

graph_file_list = train_positive_files + train_negative_files + test_positive_files + test_negative_files

#graph_file_list = [i for i in os.listdir(graphs_directory) if os.path.splitext(i)[1] == ".graphml"]

g = gt.load_graph(graph_file_list[1][0]) # Loads the first graph, to extract the list of properties (which should be the same for all graphs)
print(g.list_properties())

def sanitise_property_name(property):
	if property == "NAME":
		return "name_x"
	else:
		return property.lower()

def sanitise_property_value(value):

	return "val_" + ''.join(filter(str.isalnum, value)).lower()


bad_vertex_properties = {"AST_PARENT_FULL_NAME", "COLUMN_NUMBER", "COLUMN_NUMBER_END", "FILENAME", "FULL_NAME", "LANGUAGE", "LINE_NUMBER", "LINE_NUMBER_END", "ROOT", "VERSION", "_graphml_vertex_id", "CODE"}
bad_edge_properties = {"_graphml_edge_id"}

good_vertex_properties = set()
good_edge_properties = set()

for file in graph_file_list:

	g = gt.load_graph(file[0])

	good_vertex_properties = good_vertex_properties | set([(p, g.vertex_properties[p].value_type()) for p in g.vertex_properties if p not in bad_vertex_properties])
	good_edge_properties = good_edge_properties | set([(p, g.edge_properties[p].value_type())  for p in g.edge_properties if p not in bad_edge_properties])	

# First we generate mode declarations and determinations

b_lines.append(":- set(test_pos, \"test.f\").\n")
b_lines.append(":- set(test_neg, \"test.n\").\n")
b_lines.append(":- set(i, 2).\n")
b_lines.append(":- modeh(1, vulnerable(+graph)).\n")
b_lines.append(":- modeb(*, has_vertex(+graph, -vertex)).\n")
b_lines.append(":- modeb(*, has_edge(+graph, -edge)).\n")
b_lines.append(":- modeb(*, ast_before(+vertex, -vertex)).\n")
# b_lines.append(":- modeb(*, ast_before(-vertex, +vertex)).\n")
b_lines.append(":- modeb(*, cfg_before(+vertex, -vertex)).\n")
# b_lines.append(":- modeb(*, cfg_before(-vertex, +vertex)).\n")
b_lines.append("\n")
b_lines.append(":- determination(vulnerable/1, has_vertex/2).\n")
b_lines.append(":- determination(vulnerable/1, has_edge/2).\n")
b_lines.append(":- determination(vulnerable/1, ast_before/2).\n")
b_lines.append(":- determination(vulnerable/1, cfg_before/2).\n")
b_lines.append("\n")
b_lines.append("cfg_before(A, C) :- cfg_edge(A, B), cfg_before(B, C).\n")
b_lines.append("cfg_before(A, B) :- cfg_edge(A, B).\n")
b_lines.append("ast_before(A, C) :- ast_edge(A, B), ast_before(B, C).\n")
b_lines.append("ast_before(A, B) :- ast_edge(A, B).\n")

str_values = {}
str_properties = []

for vp in good_vertex_properties:
	if vp[1] == "int32_t":
		b_lines.append(":- modeb(1, " + sanitise_property_name(vp[0]) + "(+vertex, #int)).\n")
		b_lines.append(":- determination(vulnerable/1, " + sanitise_property_name(vp[0]) + "/2).\n")
	elif vp[1] == "string":
		str_properties.append(vp)
		str_values[sanitise_property_name(vp[0])] = {}
		b_lines.append(":- modeb(1, " + sanitise_property_name(vp[0]) + "(+vertex, #" + sanitise_property_name(vp[0]) + "_value)).\n")
		b_lines.append(":- determination(vulnerable/1, " + sanitise_property_name(vp[0]) + "/2).\n")
		# b_lines.append(":- modeb(1, " + sanitise_property_name(vp[0]) + "(+vertex, #string)).\n")
		# b_lines.append(":- determination(vulnerable/1, " + sanitise_property_name(vp[0]) + "/2).\n")
	elif vp[1] == "bool":
		b_lines.append(":- modeb(1, " + sanitise_property_name(vp[0]) + "(+vertex)).\n")
		b_lines.append(":- determination(vulnerable/1, " + sanitise_property_name(vp[0]) + "/1).\n")

keep = []
for file in graph_file_list:

	if random.random() < 1:
		keep.append(True)

		g = gt.load_graph(file[0])

		for v in g.vertices():
			for sp in str_properties:
				
				if sp[0] in g.vertex_properties:
					val = sanitise_property_name(sp[0]) + "_" + sanitise_property_value(g.vertex_properties[sp[0]][v])

					if val not in str_values[sanitise_property_name(sp[0])]:
						str_values[sanitise_property_name(sp[0])][val] = 1
					else:
						str_values[sanitise_property_name(sp[0])][val] += 1

	else:
		keep.append(False)

threshold = 2
for sp in str_properties:
	below_threshold = []
	for val in str_values[sanitise_property_name(sp[0])]:

		if str_values[sanitise_property_name(sp[0])][val] < threshold:
			below_threshold.append(val)
		else:
			b_lines.append(sanitise_property_name(sp[0]) + "_value(" + val + ").\n")

	for val in below_threshold:
		del str_values[sanitise_property_name(sp[0])][val]
	b_lines.append(sanitise_property_name(sp[0]) + "_value(" + sanitise_property_name(sp[0]) + "_other).\n")

	


b_lines.append("\n")


# Here we generate the meat of the program by encoding each of the graphs
num_files = len(graph_file_list)
file_index = 0
for f_tuple in graph_file_list:

	if keep[file_index]:

		file = f_tuple[0]
		pos = f_tuple[1]
		train = f_tuple[2]

		g = gt.load_graph(file)

		filename = os.path.splitext(os.path.basename(file))[0]

		b_lines.append("graph(graph_" + str(filename) + ").\n")

		if train:
			if not pos:
				n_lines.append("vulnerable(graph_" + str(filename) + ").\n")
			else:
				p_lines.append("vulnerable(graph_" + str(filename) + ").\n")
		else:
			if not pos:
				tn_lines.append("vulnerable(graph_" + str(filename) + ").\n")
			else:
				tp_lines.append("vulnerable(graph_" + str(filename) + ").\n")

		for v in g.vertices():
			identifier = "graph_" + str(filename) + "_vertex_" + str(v)
			b_lines.append("vertex(" + identifier + ").\n")
			b_lines.append("has_vertex(graph_" + str(filename) + ", " + identifier + ").\n")

			for vp in g.vertex_properties:
				if vp not in bad_vertex_properties:
					val = g.vertex_properties[vp][v]
					if g.vertex_properties[vp].value_type() == "int32_t":
						b_lines.append(sanitise_property_name(vp) + "(" + identifier + ", " + str(val) + ").\n")
					elif g.vertex_properties[vp].value_type() == "string":
						if val != "":
							#b_lines.append(sanitise_property_name(vp) + "(" + identifier + ", '" + val.replace("'", "\\'") + "').\n")
							sanitised_val = sanitise_property_name(vp) + "_" + sanitise_property_value(val)

							if sanitised_val in str_values[sanitise_property_name(vp)]:
								b_lines.append(sanitise_property_name(vp) + "(" + identifier + ", " + sanitised_val + ").\n")
							else:
								b_lines.append(sanitise_property_name(vp) + "(" + identifier + ", " + sanitise_property_name(vp) + "_other).\n")

					elif g.vertex_properties[vp].value_type() == "bool":
						if val == 1:
							b_lines.append(sanitise_property_name(vp) + "(" + identifier + ").\n")

			b_lines.append("\n")

		for e in g.edges():
			
			if g.edge_properties["labelE"][e] == "AST":
				b_lines.append("ast_edge(" + "graph_" + str(filename) + "_vertex_" + str(e.source()) + ", " + "graph_" + str(filename) + "_vertex_" + str(e.target()) + ").\n")
			elif g.edge_properties["labelE"][e] == "CFG":
				b_lines.append("cfg_edge(" + "graph_" + str(filename) + "_vertex_" + str(e.source()) + ", " + "graph_" + str(filename) + "_vertex_" + str(e.target()) + ").\n")


	file_index += 1

	print(str(file_index) + " / " + str(num_files))



background.writelines(b_lines)
background.close()

positive.writelines(p_lines)
positive.close()

negative.writelines(n_lines)
negative.close()

test_positive.writelines(tp_lines)
test_positive.close()

test_negative.writelines(tn_lines)
test_negative.close()