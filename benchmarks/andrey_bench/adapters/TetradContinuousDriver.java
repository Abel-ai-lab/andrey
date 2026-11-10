import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DelimiterType;
import edu.cmu.tetrad.data.DoubleDataBox;
import edu.cmu.tetrad.data.Knowledge;
import edu.cmu.tetrad.data.SimpleDataLoader;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.Boss;
import edu.cmu.tetrad.search.BossFci;
import edu.cmu.tetrad.search.Fas;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.Fges;
import edu.cmu.tetrad.search.Grasp;
import edu.cmu.tetrad.search.GraspFci;
import edu.cmu.tetrad.search.Pc;
import edu.cmu.tetrad.search.PermutationSearch;
import edu.cmu.tetrad.search.StarFci;
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
 * Runs Tetrad's PC, FGES, BOSS, GRaSP, FCI, BOSS-FCI, and GRaSP-FCI on a CSV of continuous columns
 * named x0..x{d-1}.
 *
 * <p>The file runs from source against tetrad-lib's shaded jar, with no build step. Every search
 * calls Tetrad's own class directly and takes the settings below from its arguments; any other
 * setting stays at Tetrad's default, which is no time lag and no max-P collider rule:
 *
 * <pre>
 * java --class-path tetrad-lib-shaded.jar TetradContinuousDriver.java search ALGORITHM DATA.csv
 *     WARMUP OUT [--knowledge FILE] [--setting VALUE ...]
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
 *        numThreads useDataOrder resetAfterBM resetAfterRS seed
 * grasp  penaltyDiscount structurePrior ruleType precomputeCovariances depth uncoveredDepth
 *        nonSingularDepth ordered useScore useRaskuttiUhler useDataOrder allowInternalRandomness
 *        numStarts seed
 * fci    alpha depth stable doPossibleDsep completeRuleSetUsed maxDiscriminatingPathLength
 *        r0ColliderRule guaranteePag excludeSelectionBias
 * bfci   alpha penaltyDiscount structurePrior ruleType precomputeCovariances numStarts useBes
 *        numThreads depth useMaxP completeRuleSetUsed maxDiscriminatingPathLength guaranteePag
 *        excludeSelectionBias
 * grasp-fci  alpha penaltyDiscount structurePrior ruleType precomputeCovariances depth
 *        singularDepth nonSingularDepth ordered useScore useRaskuttiUhler useDataOrder numStarts
 *        seed useMaxP completeRuleSetUsed maxDiscriminatingPathLength guaranteePag
 *        excludeSelectionBias
 * </pre>
 *
 * <p>{@code bfci} runs {@code BossFci} and {@code grasp-fci} {@code GraspFci}. Their FCI steps come
 * from {@code StarFci}, which in 7.6.10 has no setting for possible-d-separation, and
 * {@code GraspFci.setDepth} sets GRaSP's depth and the cap on the FCI steps' separating sets alike.
 *
 * <p>{@code --knowledge FILE}, which every search but grasp takes, is Tetrad's knowledge file
 * ({@code addtemporal} tiers, {@code forbiddirect} and {@code requiredirect} edges), read by
 * Tetrad's own reader as causal-cmd reads it, before the clock; without it a search has no
 * background knowledge, and grasp stops on it.
 *
 * <p>Tetrad 7.6.10 wraps the test of {@code Fci} in a cache that is a plain {@code HashMap}, and
 * runs its stable skeleton search in a parallel stream over that test, which on several threads
 * throws {@code ClassCastException} or never returns. The driver gives the skeleton search the
 * uncached test; the later steps of the search still use the cache, in one thread.
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
        int from = 4;
        Knowledge knowledge = null;
        if (args.length > from + 1 && args[from].equals("--knowledge")) {
            knowledge = SimpleDataLoader.loadKnowledge(
                    Path.of(args[from + 1]).toFile(), DelimiterType.WHITESPACE, "//");
            from += 2;
        }
        Graph graph = null;
        double seconds = 0;
        for (int i = 0; i <= warmup; i++) {
            Settings settings = new Settings(args, from);
            Knowledge copy = knowledge == null ? null : knowledge.copy();
            long start = System.nanoTime();
            graph = run(algorithm, data, settings, copy);
            seconds = (System.nanoTime() - start) / 1e9;
            settings.done();
        }
        Files.writeString(out, "Search seconds: " + seconds + "\n\n" + graph + "\n");
    }

    static Graph run(String algorithm, DataSet data, Settings s, Knowledge knowledge)
            throws Exception {
        if (knowledge != null && algorithm.equals("grasp")) {
            throw new IllegalArgumentException(algorithm + " takes no knowledge here");
        }
        switch (algorithm) {
            case "pc": {
                Pc pc = new Pc(new IndTestFisherZ(data, s.real("alpha")));
                pc.setDepth(s.integer("depth"));
                pc.setFasStable(s.bool("fasStable"));
                pc.setColliderOrientationStyle(
                        Pc.ColliderOrientationStyle.valueOf(upper(s.text("colliderOrientationStyle"))));
                pc.setAllowBidirected(s.bool("allowBidirected") ? Pc.AllowBidirected.ALLOW
                        : Pc.AllowBidirected.DISALLOW);
                if (knowledge != null) pc.setKnowledge(knowledge);
                return pc.search();
            }
            case "fges": {
                Fges fges = new Fges(score(data, s));
                fges.setMaxDegree(s.integer("maxDegree"));
                fges.setNumThreads(s.integer("numThreads"));
                fges.setSymmetricFirstStep(s.bool("symmetricFirstStep"));
                fges.setFaithfulnessAssumed(s.bool("faithfulnessAssumed"));
                if (knowledge != null) fges.setKnowledge(knowledge);
                fges.setVerbose(false);
                return fges.search();
            }
            case "boss": {
                Boss boss = new Boss(score(data, s));
                boss.setUseBes(s.bool("useBes"));
                boss.setNumStarts(s.integer("numStarts"));
                boss.setNumThreads(s.integer("numThreads"));
                boss.setUseDataOrder(s.bool("useDataOrder"));
                boss.setResetAfterBM(s.bool("resetAfterBM"));
                boss.setResetAfterRS(s.bool("resetAfterRS"));
                boss.setVerbose(false);
                PermutationSearch search = new PermutationSearch(boss);
                if (knowledge != null) search.setKnowledge(knowledge);
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
                IndTestFisherZ test = new IndTestFisherZ(data, s.real("alpha"));
                Fci fci = new Fci(test);
                fci.setDepth(s.integer("depth"));
                fci.setStable(s.bool("stable"));
                fci.setDoPossibleDsep(s.bool("doPossibleDsep"));
                fci.setCompleteRuleSetUsed(s.bool("completeRuleSetUsed"));
                fci.setMaxDiscriminatingPathLength(s.integer("maxDiscriminatingPathLength"));
                fci.setR0ColliderRule(Fci.ColliderRule.valueOf(upper(s.text("r0ColliderRule"))));
                fci.setGuaranteePag(s.bool("guaranteePag"));
                fci.setExcludeSelectionBias(s.bool("excludeSelectionBias"));
                if (knowledge != null) fci.setKnowledge(knowledge);
                fci.setVerbose(false);
                return fci.search(new Fas(test));
            }
            case "bfci": {
                BossFci bfci =
                        new BossFci(new IndTestFisherZ(data, s.real("alpha")), score(data, s));
                bfci.setNumStarts(s.integer("numStarts"));
                bfci.setBossUseBes(s.bool("useBes"));
                bfci.setNumThreads(s.integer("numThreads"));
                bfci.setDepth(s.integer("depth"));
                return starFci(bfci, s, knowledge);
            }
            case "grasp-fci": {
                GraspFci graspFci =
                        new GraspFci(new IndTestFisherZ(data, s.real("alpha")), score(data, s));
                graspFci.setDepth(s.integer("depth"));
                graspFci.setSingularDepth(s.integer("singularDepth"));
                graspFci.setNonSingularDepth(s.integer("nonSingularDepth"));
                graspFci.setOrdered(s.bool("ordered"));
                graspFci.setUseScore(s.bool("useScore"));
                graspFci.setUseRaskuttiUhler(s.bool("useRaskuttiUhler"));
                graspFci.setUseDataOrder(s.bool("useDataOrder"));
                graspFci.setNumStarts(s.integer("numStarts"));
                graspFci.setSeed(s.integer("seed"));
                return starFci(graspFci, s, knowledge);
            }
            default:
                throw new IllegalArgumentException("unknown algorithm " + algorithm);
        }
    }

    static String upper(String name) {
        return name.toUpperCase(Locale.ROOT);
    }

    /** The FCI steps BOSS-FCI and GRaSP-FCI share, set from {@code s}; then the search. */
    static Graph starFci(StarFci search, Settings s, Knowledge knowledge) throws Exception {
        search.setUseMaxP(s.bool("useMaxP"));
        search.setCompleteRuleSetUsed(s.bool("completeRuleSetUsed"));
        search.setMaxDiscriminatingPathLength(s.integer("maxDiscriminatingPathLength"));
        search.setGuaranteePag(s.bool("guaranteePag"));
        search.setExcludeSelectionBias(s.bool("excludeSelectionBias"));
        if (knowledge != null) search.setKnowledge(knowledge);
        search.setVerbose(false);
        return search.search();
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
