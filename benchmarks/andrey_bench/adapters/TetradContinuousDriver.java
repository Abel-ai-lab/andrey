import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DoubleDataBox;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.Boss;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.Fges;
import edu.cmu.tetrad.search.Grasp;
import edu.cmu.tetrad.search.Pc;
import edu.cmu.tetrad.search.PermutationSearch;
import edu.cmu.tetrad.search.score.SemBicScore;
import edu.cmu.tetrad.search.test.IndTestFisherZ;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Runs Tetrad's PC, FGES, BOSS, GRaSP, and FCI on a CSV of continuous columns named x0..x{d-1}.
 *
 * <p>The file runs from source against tetrad-lib's shaded jar, with no build step. Every search
 * calls Tetrad's own class directly and takes every setting from its arguments, none from a
 * default:
 *
 * <pre>
 * java --class-path tetrad-lib-shaded.jar TetradContinuousDriver.java search ALGORITHM DATA.csv
 *     WARMUP OUT [--setting VALUE ...]
 * java --class-path tetrad-lib-shaded.jar TetradContinuousDriver.java ptest DATA.csv QUERIES
 * java --class-path tetrad-lib-shaded.jar TetradContinuousDriver.java score DATA.csv PENALTY QUERIES
 * </pre>
 *
 * <p>{@code search} runs WARMUP untimed searches first in the same JVM; the timed search builds the
 * test or score from the data and runs the search, and nothing else. OUT gets {@code Search
 * seconds:} and the timed search's seconds, a blank line, and the graph in Tetrad's text form.
 * The settings are named for the Tetrad setter they reach, and a missing or unknown one stops the
 * run:
 *
 * <pre>
 * pc     alpha depth fasStable colliderOrientationStyle allowBidirected
 * fges   penaltyDiscount structurePrior ruleType precomputeCovariances maxDegree numThreads
 *        symmetricFirstStep faithfulnessAssumed
 * boss   penaltyDiscount structurePrior ruleType precomputeCovariances useBes numStarts
 *        numThreads useDataOrder seed
 * grasp  penaltyDiscount structurePrior ruleType precomputeCovariances depth uncoveredDepth
 *        nonSingularDepth ordered useScore useRaskuttiUhler useDataOrder allowInternalRandomness
 *        numStarts seed
 * fci    alpha depth stable doPossibleDsep completeRuleSetUsed maxDiscriminatingPathLength
 *        r0ColliderRule guaranteePag excludeSelectionBias
 * </pre>
 *
 * <p>{@code ptest} prints the Fisher z p-value of each "x y z1 ..." line of QUERIES (column
 * indices), and {@code score} the local BIC score of each "node p1 p2 ..." line at PENALTY, the
 * quantities the settings above reach.
 */
public class TetradContinuousDriver {
    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "";
        switch (mode) {
            case "search" -> search(Arrays.copyOfRange(args, 1, args.length));
            case "ptest" -> ptest(read(Path.of(args[1])), Path.of(args[2]));
            case "score" -> score(read(Path.of(args[1])), Double.parseDouble(args[2]),
                    Path.of(args[3]));
            default -> {
                System.err.println("usage: TetradContinuousDriver.java MODE ARGS; MODE is search,"
                        + " ptest, or score");
                System.exit(2);
            }
        }
    }

    /** The arguments after the fixed ones, as {@code --name value} pairs read once each. */
    static final class Settings {
        private final Map<String, String> values = new HashMap<>();
        private final Set<String> read = new HashSet<>();

        Settings(String[] args, int from) {
            for (int i = from; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || i + 1 >= args.length) {
                    throw new IllegalArgumentException("expected --name value, got " + args[i]);
                }
                values.put(args[i].substring(2), args[i + 1]);
            }
        }

        String text(String name) {
            String value = values.get(name);
            if (value == null) {
                throw new IllegalArgumentException("missing setting --" + name);
            }
            read.add(name);
            return value;
        }

        double real(String name) { return Double.parseDouble(text(name)); }

        int integer(String name) { return Integer.parseInt(text(name)); }

        boolean bool(String name) { return Boolean.parseBoolean(text(name)); }

        /** Stops the run on a setting the algorithm never read. */
        void done() {
            for (String name : values.keySet()) {
                if (!read.contains(name)) {
                    throw new IllegalArgumentException("unknown setting --" + name);
                }
            }
        }
    }

    static void search(String[] args) throws Exception {
        String algorithm = args[0];
        DataSet data = read(Path.of(args[1]));
        int warmup = Integer.parseInt(args[2]);
        Path out = Path.of(args[3]);
        Graph graph = null;
        double seconds = 0;
        for (int i = 0; i <= warmup; i++) {
            Settings settings = new Settings(args, 4);
            long start = System.nanoTime();
            graph = run(algorithm, data, settings);
            seconds = (System.nanoTime() - start) / 1e9;
            settings.done();
        }
        Files.writeString(out, "Search seconds: " + seconds + "\n\n" + graph + "\n");
    }

    static Graph run(String algorithm, DataSet data, Settings s) throws Exception {
        switch (algorithm) {
            case "pc": {
                Pc pc = new Pc(new IndTestFisherZ(data, s.real("alpha")));
                pc.setDepth(s.integer("depth"));
                pc.setFasStable(s.bool("fasStable"));
                pc.setColliderOrientationStyle(
                        Pc.ColliderOrientationStyle.valueOf(upper(s.text("colliderOrientationStyle"))));
                pc.setAllowBidirected(s.bool("allowBidirected") ? Pc.AllowBidirected.ALLOW
                        : Pc.AllowBidirected.DISALLOW);
                return pc.search();
            }
            case "fges": {
                Fges fges = new Fges(score(data, s));
                fges.setMaxDegree(s.integer("maxDegree"));
                fges.setNumThreads(s.integer("numThreads"));
                fges.setSymmetricFirstStep(s.bool("symmetricFirstStep"));
                fges.setFaithfulnessAssumed(s.bool("faithfulnessAssumed"));
                fges.setVerbose(false);
                return fges.search();
            }
            case "boss": {
                Boss boss = new Boss(score(data, s));
                boss.setUseBes(s.bool("useBes"));
                boss.setNumStarts(s.integer("numStarts"));
                boss.setNumThreads(s.integer("numThreads"));
                boss.setUseDataOrder(s.bool("useDataOrder"));
                boss.setVerbose(false);
                PermutationSearch search = new PermutationSearch(boss);
                search.setSeed(s.integer("seed"));
                return search.search(true);
            }
            case "grasp": {
                SemBicScore score = score(data, s);
                Grasp grasp = new Grasp(score);
                grasp.setDepth(s.integer("depth"));
                grasp.setUncoveredDepth(s.integer("uncoveredDepth"));
                grasp.setNonSingularDepth(s.integer("nonSingularDepth"));
                grasp.setOrdered(s.bool("ordered"));
                grasp.setUseScore(s.bool("useScore"));
                grasp.setUseRaskuttiUhler(s.bool("useRaskuttiUhler"));
                grasp.setUseDataOrder(s.bool("useDataOrder"));
                grasp.setAllowInternalRandomness(s.bool("allowInternalRandomness"));
                grasp.setNumStarts(s.integer("numStarts"));
                grasp.setSeed(s.integer("seed"));
                grasp.setVerbose(false);
                grasp.bestOrder(score.getVariables());
                return grasp.getGraph(true, false);
            }
            case "fci": {
                Fci fci = new Fci(new IndTestFisherZ(data, s.real("alpha")));
                fci.setDepth(s.integer("depth"));
                fci.setStable(s.bool("stable"));
                fci.setDoPossibleDsep(s.bool("doPossibleDsep"));
                fci.setCompleteRuleSetUsed(s.bool("completeRuleSetUsed"));
                fci.setMaxDiscriminatingPathLength(s.integer("maxDiscriminatingPathLength"));
                fci.setR0ColliderRule(Fci.ColliderRule.valueOf(upper(s.text("r0ColliderRule"))));
                fci.setGuaranteePag(s.bool("guaranteePag"));
                fci.setExcludeSelectionBias(s.bool("excludeSelectionBias"));
                fci.setVerbose(false);
                return fci.search();
            }
            default:
                throw new IllegalArgumentException("unknown algorithm " + algorithm);
        }
    }

    static String upper(String name) {
        return name.toUpperCase(Locale.ROOT);
    }

    /** Tetrad's linear-Gaussian BIC from the score settings the three score searches share. */
    static SemBicScore score(DataSet data, Settings s) {
        SemBicScore score = new SemBicScore(data, s.bool("precomputeCovariances"));
        score.setPenaltyDiscount(s.real("penaltyDiscount"));
        score.setStructurePrior(s.real("structurePrior"));
        score.setRuleType(SemBicScore.RuleType.valueOf(upper(s.text("ruleType"))));
        return score;
    }

    static void ptest(DataSet data, Path queries) throws Exception {
        IndTestFisherZ test = new IndTestFisherZ(data, 0.05);
        List<Node> vars = test.getVariables();
        for (int[] q : rows(queries)) {
            Set<Node> given = new HashSet<>();
            for (int i = 2; i < q.length; i++) given.add(vars.get(q[i]));
            double p = test.getPValue(vars.get(q[0]), vars.get(q[1]), given);
            System.out.println(line(q) + " p=" + p);
        }
    }

    static void score(DataSet data, double penalty, Path queries) throws Exception {
        SemBicScore score = new SemBicScore(data, true);
        score.setPenaltyDiscount(penalty);
        score.setStructurePrior(0);
        score.setRuleType(SemBicScore.RuleType.CHICKERING);
        for (int[] q : rows(queries)) {
            System.out.println(line(q) + " score=" + score.localScore(q[0],
                    Arrays.copyOfRange(q, 1, q.length)));
        }
    }

    /** Whitespace-separated integer rows of a file; blank lines skipped. */
    static List<int[]> rows(Path path) throws Exception {
        List<int[]> rows = new ArrayList<>();
        for (String text : Files.readAllLines(path)) {
            if (!text.isBlank()) {
                rows.add(Arrays.stream(text.trim().split("\\s+")).mapToInt(Integer::parseInt)
                        .toArray());
            }
        }
        return rows;
    }

    static String line(int[] q) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < q.length; i++) text.append(i > 0 ? " " : "").append(q[i]);
        return text.toString();
    }

    static DataSet read(Path csv) throws Exception {
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
