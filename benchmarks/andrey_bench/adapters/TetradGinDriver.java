import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DoubleDataBox;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
import edu.cmu.tetrad.search.Gin;
import edu.cmu.tetrad.search.test.Kci;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Runs Tetrad's GIN on a CSV of continuous columns named x0..x{d-1} and writes its clusters.
 *
 * <pre>java --class-path tetrad-lib-shaded.jar TetradGinDriver.java DATA.csv ALPHA WARMUP OUT</pre>
 *
 * The test is Tetrad's KCI with its defaults (Gaussian kernel, median-distance bandwidth, gamma
 * approximation), the raw marginal test GIN takes. WARMUP untimed searches run first in the same
 * JVM; the timed search is the {@code Gin.search} call alone. OUT gets three lines:
 * {@code Search seconds:}, then {@code Ordered clusters:} and {@code Unordered clusters:}, each a
 * {@code ;}-separated list of clusters given as space-separated column indices. Tetrad names its
 * latents L1, L2, ... in its causal order and joins the latents it could not order by undirected
 * edges, so a latent with an undirected edge is unordered.
 */
public class TetradGinDriver {
    public static void main(String[] args) throws Exception {
        Path csv = Path.of(args[0]);
        double alpha = Double.parseDouble(args[1]);
        int warmup = Integer.parseInt(args[2]);
        Path out = Path.of(args[3]);

        DataSet data = read(csv);
        for (int i = 0; i < warmup; i++) {
            search(data, alpha);
        }
        long start = System.nanoTime();
        Graph graph = search(data, alpha);
        double seconds = (System.nanoTime() - start) / 1e9;

        List<Node> latents = graph.getNodes().stream()
                .filter(n -> n.getNodeType() == NodeType.LATENT)
                .sorted(Comparator.comparingInt(n -> Integer.parseInt(n.getName().substring(1))))
                .collect(Collectors.toList());
        List<String> ordered = new ArrayList<>();
        List<String> unordered = new ArrayList<>();
        for (Node latent : latents) {
            String members = graph.getChildren(latent).stream()
                    .filter(c -> c.getNodeType() != NodeType.LATENT)
                    .map(c -> Integer.parseInt(c.getName().substring(1)))
                    .sorted()
                    .map(String::valueOf)
                    .collect(Collectors.joining(" "));
            boolean free = false;
            for (Edge edge : graph.getEdges(latent)) {
                free |= Edges.isUndirectedEdge(edge);
            }
            (free ? unordered : ordered).add(members);
        }
        Files.writeString(out, "Search seconds: " + seconds + "\n"
                + "Ordered clusters: " + String.join(";", ordered) + "\n"
                + "Unordered clusters: " + String.join(";", unordered) + "\n");
    }

    private static Graph search(DataSet data, double alpha) {
        Kci test = new Kci(data);
        test.setAlpha(alpha);
        return new Gin(alpha, test).search(data);
    }

    private static DataSet read(Path csv) throws Exception {
        List<String> lines = Files.readAllLines(csv);
        String[] names = lines.get(0).split(",");
        List<Node> nodes = new ArrayList<>();
        for (String name : names) {
            nodes.add(new ContinuousVariable(name.trim()));
        }
        double[][] rows = new double[lines.size() - 1][names.length];
        for (int i = 1; i < lines.size(); i++) {
            String[] cells = lines.get(i).split(",");
            for (int j = 0; j < names.length; j++) {
                rows[i - 1][j] = Double.parseDouble(cells[j]);
            }
        }
        return new BoxDataSet(new DoubleDataBox(rows), nodes);
    }
}
