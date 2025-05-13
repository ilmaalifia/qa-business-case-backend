import unittest

from app.graph import graph as compiled_graph


class TestGraphStructure(unittest.TestCase):
    def test_graph_structure(self):
        graph = compiled_graph.get_graph()
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)

        i = 0
        nodes_sequence = ["__start__", "retriever", "generator", "__end__"]
        for k, v in graph.nodes.items():
            self.assertEqual(k, nodes_sequence[i])
            self.assertEqual(v.id, nodes_sequence[i])
            i += 1
