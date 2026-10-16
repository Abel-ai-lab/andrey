import edu.cmu.tetrad.algcomparison.algorithm.Algorithm;
import edu.cmu.tetrad.algcomparison.algorithm.AlgorithmFactory;
import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.IndependenceFact;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.Fges;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.search.Pc;
import edu.cmu.tetrad.search.score.ConditionalGaussianScore;
import edu.cmu.tetrad.search.score.DegenerateGaussianScore;
import edu.cmu.tetrad.search.score.Score;
import edu.cmu.tetrad.search.test.IndTestConditionalGaussianLrt;
import edu.cmu.tetrad.search.test.IndTestDegenerateGaussianLrt;
import edu.cmu.tetrad.search.test.IndependenceResult;
import edu.cmu.tetrad.search.utils.PcCommon;
import edu.cmu.tetrad.search.utils.SepsetMap;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.Parameters;
import edu.pitt.dbmi.causal.cmd.CmdArgs;
import edu.pitt.dbmi.causal.cmd.CmdParser;
import edu.pitt.dbmi.causal.cmd.data.DataFiles;
import edu.pitt.dbmi.causal.cmd.tetrad.Tetrad;
import edu.pitt.dbmi.data.reader.DataColumn;
import edu.pitt.dbmi.data.reader.tabular.TabularColumnFileReader;
import edu.pitt.dbmi.data.reader.tabular.TabularColumnReader;
import edu.pitt.dbmi.data.reader.tabular.TabularDataFileReader;
import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Runs Tetrad's searches as causal-cmd 1.12.0 does, with the categorical columns declared.
 *
 * <p>causal-cmd 1.12.0 parses {@code --numCategories} but never stores it, so its mixed-data reader
 * marks a column categorical only when the column holds at most 0 distinct values: it reads every
 * column as continuous. The {@code search} mode takes causal-cmd's own argument parser, parameters,
 * and algorithm factory from the jar and replaces only that reader with one that marks a column
 * categorical when it holds at most {@code --numCategories} distinct values. The file runs from
 * source, with no build step:
 *
 * <pre>
 * java --class-path causal-cmd-1.12.0-jar-with-dependencies.jar TetradDriver.java search ARGS
 * </pre>
 *
 * <p>Modes ({@code k} is the most distinct values of a categorical column; columns by index):
 *
 * <pre>
 * search [--warmup N] ARGS    run causal-cmd's algorithm on causal-cmd's arguments ARGS, after N
 *                             untimed runs (default 0) in the same JVM; write OUT/PREFIX_out.txt:
 *                             "Categorical columns: " and their names joined by ";", "Search
 *                             seconds: " and the timed search call's wall time, a blank line,
 *                             then the graph
 * columns CSV k               each column's type as causal-cmd's reader and as this reader read it
 * pc CSV k cg|dg [TABLE]      PC as causal-cmd sets it up; print every test, the separating sets,
 *                             and the graph; with TABLE ("x y z1 ... p" lines), the p-values come
 *                             from it, and a query it lacks exits 3
 * ptest CSV k cg|dg QUERIES   the p-value of each "x y z1 ..." line
 * score CSV k cg|dg QUERIES   the local score of each "node p1 p2 ..." line
 * fges CSV k cg|dg            FGES as causal-cmd sets it up; print the graph and its score
 * </pre>
 */
public class TetradDriver {
    static final PrintStream QUIET = new PrintStream(OutputStream.nullOutputStream());

    public static void main(String[] argv) throws Exception {
        String mode = argv.length > 0 ? argv[0] : "";
        String[] rest = Arrays.copyOfRange(argv, Math.min(1, argv.length), argv.length);
        switch (mode) {
            case "search" -> search(rest);
            case "columns" -> columns(rest[0], Integer.parseInt(rest[1]));
            case "pc", "ptest", "score", "fges" -> Probe.run(mode, rest);
            default -> {
                System.err.println("usage: TetradDriver.java MODE ARGS; MODE is search, columns,"
                        + " pc, ptest, score, or fges");
                System.exit(2);
            }
        }
        System.out.flush();
    }

    /** The value of {@code --numCategories} in causal-cmd's arguments. */
    static int numCategories(String[] args) {
        int at = Arrays.asList(args).indexOf("--numCategories");
        if (at < 0 || at + 1 >= args.length) {
            throw new IllegalArgumentException("--numCategories is required");
        }
        return Integer.parseInt(args[at + 1]);
    }

    /** causal-cmd's tabular reader for one file, with the categorical columns declared. */
    static DataSet read(CmdArgs args, int numCategories) throws Exception {
        if (!args.isHasHeader() || args.getDatasetFiles().size() != 1) {
            throw new IllegalArgumentException("expects one data file with a header");
        }
        Path file = args.getDatasetFiles().get(0);
        TabularColumnReader columnReader = new TabularColumnFileReader(file, args.getDelimiter());
        columnReader.setCommentMarker(args.getCommentMarker());
        columnReader.setQuoteCharacter(args.getQuoteChar());
        boolean discrete = args.getDataType() == DataType.Discrete;
        DataColumn[] columns = columnReader.readInDataColumns(new HashSet<>(), discrete);
        if (args.getDataType() == DataType.Mixed) {
            TabularDataReader typer = new TabularDataFileReader(file, args.getDelimiter());
            typer.setCommentMarker(args.getCommentMarker());
            typer.setQuoteCharacter(args.getQuoteChar());
            typer.setMissingDataMarker(args.getMissingValueMarker());
            typer.determineDiscreteDataColumns(columns, numCategories, true);
        }
        TabularDataReader reader = new TabularDataFileReader(file, args.getDelimiter());
        reader.setCommentMarker(args.getCommentMarker());
        reader.setQuoteCharacter(args.getQuoteChar());
        reader.setMissingDataMarker(args.getMissingValueMarker());
        return (DataSet) DataConvertUtils.toDataModel(reader.read(columns, true, null));
    }

    /**
     * Runs the search {@code --warmup} times, then once more, and keeps the last run's graph and
     * time. The clock holds the search call alone: the JVM start, the compile of this file, and
     * the data read come before it.
     */
    @SuppressWarnings("unchecked")
    static void search(String[] argv) throws Exception {
        int warmup = 0;
        if (argv.length > 1 && argv[0].equals("--warmup")) {
            warmup = Integer.parseInt(argv[1]);
            argv = Arrays.copyOfRange(argv, 2, argv.length);
        }
        CmdArgs args = CmdParser.parse(argv);
        DataSet data = read(args, numCategories(argv));
        Graph graph = null;
        double seconds = 0;
        for (int i = 0; i <= warmup; i++) {
            Algorithm algorithm = AlgorithmFactory.create(
                    args.getAlgorithmClass(), args.getTestClass(), args.getScoreClass());
            Parameters parameters = Tetrad.getParameters(args);
            parameters.set("printStream", QUIET);
            long start = System.nanoTime();
            graph = algorithm.search(data, parameters);
            seconds = (System.nanoTime() - start) / 1e9;
        }
        StringJoiner categorical = new StringJoiner(";");
        for (Node v : data.getVariables()) {
            if (v instanceof DiscreteVariable) categorical.add(v.getName());
        }
        Path out = args.getOutDirectory().resolve(args.getFilePrefix() + "_out.txt");
        try (PrintStream w = new PrintStream(Files.newOutputStream(out))) {
            w.println("Categorical columns: " + categorical);
            w.println("Search seconds: " + seconds);
            w.println();
            w.println(graph);
        }
    }

    /** causal-cmd's parsed arguments for a mixed-data PC run on {@code csv}, for its defaults. */
    static CmdArgs defaults(String csv, int numCategories) throws Exception {
        return CmdParser.parse(new String[]{"--data-type", "mixed", "--numCategories",
                String.valueOf(numCategories), "--dataset", csv, "--delimiter", "comma",
                "--algorithm", "pc", "--test", "cg-lr-test",
                "--out", System.getProperty("java.io.tmpdir")});
    }

    static void columns(String csv, int numCategories) throws Exception {
        CmdArgs args = defaults(csv, numCategories);
        DataSet theirs = (DataSet) DataFiles.readInDatasets(args, QUIET).get(0);
        DataSet ours = read(args, numCategories);
        for (int j = 0; j < ours.getNumColumns(); j++) {
            System.out.println("COLUMN " + j + " " + ours.getVariable(j).getName() + " causal-cmd="
                    + kind(theirs.getVariable(j)) + " declared=" + kind(ours.getVariable(j)));
        }
        int differ = 0;
        for (int i = 0; i < ours.getNumRows(); i++) {
            for (int j = 0; j < ours.getNumColumns(); j++) {
                double b = ours.getVariable(j) instanceof DiscreteVariable v
                        ? Double.parseDouble(v.getCategory(ours.getInt(i, j)))
                        : ours.getDouble(i, j);
                if (theirs.getDouble(i, j) != b) differ++;
            }
        }
        System.out.println("cells differing: " + differ);
    }

    static String kind(Node v) {
        return v instanceof DiscreteVariable c ? "categorical" + c.getCategories() : "continuous";
    }

    /** Tetrad's tests, scores, PC, and FGES called directly, set up as causal-cmd sets them up. */
    static final class Probe {
        static void run(String mode, String[] a) throws Exception {
            DataSet ds = read(defaults(a[0], Integer.parseInt(a[1])), Integer.parseInt(a[1]));
            List<Node> vars = ds.getVariables();
            String kind = a[2];
            PrintStream out = System.out;
            switch (mode) {
                case "pc" -> pc(ds, kind, a.length > 3 ? table(a[3]) : null, out);
                case "ptest" -> {
                    IndependenceTest t = test(ds, kind);
                    for (int[] q : rows(a[3])) {
                        Set<Node> z = new LinkedHashSet<>();
                        for (int i = 2; i < q.length; i++) z.add(vars.get(q[i]));
                        IndependenceResult r =
                                t.checkIndependence(vars.get(q[0]), vars.get(q[1]), z);
                        out.println(line(q) + " p=" + r.getPValue()
                                + " indep=" + r.isIndependent());
                    }
                }
                case "score" -> {
                    Score s = score(ds, kind);
                    for (int[] q : rows(a[3])) {
                        int[] parents = Arrays.copyOfRange(q, 1, q.length);
                        out.println(line(q) + " score=" + s.localScore(q[0], parents));
                    }
                }
                case "fges" -> {
                    Fges f = new Fges(score(ds, kind));
                    f.setMaxDegree(1000);
                    f.setSymmetricFirstStep(false);
                    f.setFaithfulnessAssumed(false);
                    f.setParallelized(false);
                    printGraph(f.search(), vars, out);
                    out.println("MODELSCORE " + f.getModelScore());
                }
                default -> throw new IllegalArgumentException(mode);
            }
        }

        static IndependenceTest test(DataSet ds, String kind) {
            if (kind.equals("cg")) return new IndTestConditionalGaussianLrt(ds, 0.05, false);
            IndTestDegenerateGaussianLrt t = new IndTestDegenerateGaussianLrt(ds);
            t.setAlpha(0.05);
            return t;
        }

        static Score score(DataSet ds, String kind) {
            if (kind.equals("cg")) {
                ConditionalGaussianScore s = new ConditionalGaussianScore(ds, 1.0, false);
                s.setNumCategoriesToDiscretize(3);
                s.setStructurePrior(0);
                return s;
            }
            DegenerateGaussianScore s = new DegenerateGaussianScore(ds, false);
            s.setPenaltyDiscount(1.0);
            return s;
        }

        static void pc(DataSet ds, String kind, Map<String, Double> table, PrintStream out) {
            List<Node> vars = ds.getVariables();
            Pc pc = new Pc(new Logged(test(ds, kind), table, out));
            pc.setDepth(-1);
            pc.setMeekPreventCycles(false);
            pc.setStable(true);
            pc.setConflictRule(PcCommon.ConflictRule.PRIORITIZE_EXISTING);
            pc.setUseMaxPHeuristic(false);
            pc.setPcHeuristicType(PcCommon.PcHeuristicType.NONE);
            Graph g = pc.search();
            SepsetMap sep = pc.getSepsets();
            for (int i = 0; i < vars.size(); i++) {
                for (int j = i + 1; j < vars.size(); j++) {
                    Set<Node> s = sep.get(vars.get(i), vars.get(j));
                    if (s != null) {
                        List<Integer> zs = new ArrayList<>();
                        for (Node n : s) zs.add(vars.indexOf(n));
                        Collections.sort(zs);
                        out.println("SEPSET " + i + " " + j + " " + zs);
                    }
                }
            }
            printGraph(g, vars, out);
        }

        /** Whitespace-separated integer rows of a file; blank lines skipped. */
        static List<int[]> rows(String path) throws Exception {
            List<int[]> rows = new ArrayList<>();
            for (String l : Files.readAllLines(Paths.get(path))) {
                if (l.isBlank()) continue;
                String[] f = l.trim().split("\\s+");
                rows.add(Arrays.stream(f).mapToInt(Integer::parseInt).toArray());
            }
            return rows;
        }

        static String line(int[] q) {
            StringJoiner j = new StringJoiner(" ");
            for (int v : q) j.add(String.valueOf(v));
            return j.toString();
        }

        /** "x y z1 ... p" lines -> p by the query's key. */
        static Map<String, Double> table(String path) throws Exception {
            Map<String, Double> table = new HashMap<>();
            for (String l : Files.readAllLines(Paths.get(path))) {
                String[] f = l.trim().split("\\s+");
                if (f.length < 3) continue;
                List<Integer> z = new ArrayList<>();
                for (int i = 2; i < f.length - 1; i++) z.add(Integer.parseInt(f[i]));
                String k = key(Integer.parseInt(f[0]), Integer.parseInt(f[1]), z);
                table.put(k, Double.parseDouble(f[f.length - 1]));
            }
            return table;
        }

        static String key(int a, int b, List<Integer> z) {
            List<Integer> zs = new ArrayList<>(z);
            Collections.sort(zs);
            StringBuilder sb = new StringBuilder();
            sb.append(Math.min(a, b)).append(' ').append(Math.max(a, b)).append(" |");
            for (int v : zs) sb.append(' ').append(v);
            return sb.toString();
        }

        static void printGraph(Graph g, List<Node> vars, PrintStream out) {
            for (Edge e : g.getEdges()) {
                out.println("EDGE " + vars.indexOf(e.getNode1()) + " " + mark(e.getEndpoint1())
                        + " " + mark(e.getEndpoint2()) + " " + vars.indexOf(e.getNode2()));
            }
        }

        static String mark(Endpoint e) {
            return e == Endpoint.ARROW ? "ARROW" : e == Endpoint.TAIL ? "TAIL" : e.toString();
        }
    }

    /** Prints every test; with a table, answers from it instead of the inner test. */
    static final class Logged implements IndependenceTest {
        final IndependenceTest inner;
        final Map<String, Double> table;
        final List<Node> vars;
        final PrintStream out;
        int count = 0;

        Logged(IndependenceTest inner, Map<String, Double> table, PrintStream out) {
            this.inner = inner;
            this.table = table;
            this.vars = inner.getVariables();
            this.out = out;
        }

        public IndependenceResult checkIndependence(Node x, Node y, Set<Node> z) {
            List<Integer> zs = new ArrayList<>();
            for (Node n : z) zs.add(vars.indexOf(n));
            String k = Probe.key(vars.indexOf(x), vars.indexOf(y), zs);
            IndependenceResult r;
            if (table != null) {
                Double p = table.get(k);
                if (p == null) {
                    out.println("MISSING " + k);
                    out.flush();
                    System.exit(3);
                }
                r = new IndependenceResult(new IndependenceFact(x, y, z), p > 0.05, p, 0.05 - p);
            } else {
                r = inner.checkIndependence(x, y, z);
            }
            out.println("TEST " + (count++) + " " + vars.indexOf(x) + " " + vars.indexOf(y)
                    + " " + k + " p=" + r.getPValue() + " indep=" + r.isIndependent());
            return r;
        }

        public List<Node> getVariables() { return inner.getVariables(); }
        public DataModel getData() { return inner.getData(); }
        public boolean isVerbose() { return false; }
        public void setVerbose(boolean v) { }
        public double getAlpha() { return 0.05; }
        public void setAlpha(double a) { }
        public String toString() { return "logged " + inner; }
    }
}
