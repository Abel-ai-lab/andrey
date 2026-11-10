import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DoubleDataBox;
import edu.cmu.tetrad.data.missing.MissingDataPolicy;
import edu.cmu.tetrad.data.missing.MissingDataSpec;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.Boss;
import edu.cmu.tetrad.search.Fges;
import edu.cmu.tetrad.search.Grasp;
import edu.cmu.tetrad.search.PermutationSearch;
import edu.cmu.tetrad.search.score.SemBicScore;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Runs Tetrad's FGES, BOSS, and GRaSP with the linear-Gaussian BIC on a CSV of continuous columns
 * named x0..x{d-1}, where a missing value is written {@code nan}, under one of Tetrad's
 * missing-data policies.
 *
 * <p>The file runs from source against tetrad-lib's shaded jar, 8.0.0 or later, the first release
 * with {@code MissingDataPolicy}; it has no build step:
 *
 * <pre>
 * java --class-path tetrad-lib-shaded.jar TetradMissingDriver.java ALGORITHM DATA.csv WARMUP
 *     OUT --setting VALUE ...
 * </pre>
 *
 * <p>WARMUP untimed searches run first in the same JVM. The timed region builds
 * {@code SemBicScore} from the data and the policy, which under {@code EM_COVARIANCE} estimates
 * the covariance by EM, and runs the search. OUT gets
 * {@code Search seconds:} and that time, a blank line, and the CPDAG in Tetrad's text form. Every
 * setting is named for the Tetrad setter or {@code MissingDataSpec} method it reaches, and a
 * missing or unknown one stops the run:
 *
 * <pre>
 * score  missingDataPolicy emRidge emTolerance emMaxIterations essMode penaltyDiscount
 *        structurePrior ruleType precomputeCovariances
 * fges   maxDegree numThreads symmetricFirstStep faithfulnessAssumed
 * boss   useBes numStarts numThreads useDataOrder resetAfterBM resetAfterRS seed
 * grasp  depth uncoveredDepth nonSingularDepth ordered useScore useRaskuttiUhler useDataOrder
 *        allowInternalRandomness numStarts
 * </pre>
 *
 * <p>{@code missingDataPolicy} is a {@code MissingDataPolicy} constant in lower case:
 * {@code em_covariance}, {@code pairwise_covariance}, {@code testwise}, {@code listwise}, or
 * {@code fail}, which stops on a missing value. The three EM settings are read under every policy
 * and used under {@code em_covariance}.
 */
public class TetradMissingDriver {
    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("usage: TetradMissingDriver.java ALGORITHM DATA.csv WARMUP OUT"
                    + " --setting VALUE ...");
            System.exit(2);
        }
        DataSet data = read(Path.of(args[1]));
        int warmup = Integer.parseInt(args[2]);
        Graph graph = null;
        double seconds = 0;
        for (int i = 0; i <= warmup; i++) {
            Settings settings = new Settings(args, 4);
            long start = System.nanoTime();
            graph = run(args[0], data, settings);
            seconds = (System.nanoTime() - start) / 1e9;
            settings.done();
        }
        Files.writeString(Path.of(args[3]), "Search seconds: " + seconds + "\n\n" + graph + "\n");
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

    static Graph run(String algorithm, DataSet data, Settings s) throws Exception {
        SemBicScore score = score(data, s);
        switch (algorithm) {
            case "fges": {
                Fges fges = new Fges(score);
                fges.setMaxDegree(s.integer("maxDegree"));
                fges.setNumThreads(s.integer("numThreads"));
                fges.setSymmetricFirstStep(s.bool("symmetricFirstStep"));
                fges.setFaithfulnessAssumed(s.bool("faithfulnessAssumed"));
                fges.setVerbose(false);
                return fges.search();
            }
            case "boss": {
                Boss boss = new Boss(score);
                boss.setUseBes(s.bool("useBes"));
                boss.setNumStarts(s.integer("numStarts"));
                boss.setNumThreads(s.integer("numThreads"));
                boss.setUseDataOrder(s.bool("useDataOrder"));
                boss.setResetAfterBM(s.bool("resetAfterBM"));
                boss.setResetAfterRS(s.bool("resetAfterRS"));
                boss.setVerbose(false);
                PermutationSearch search = new PermutationSearch(boss);
                search.setSeed(s.integer("seed"));
                return search.search(true);
            }
            case "grasp": {
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
                grasp.setVerbose(false);
                grasp.bestOrder(score.getVariables());
                return grasp.getGraph(true);
            }
            default:
                throw new IllegalArgumentException("unknown algorithm " + algorithm);
        }
    }

    static String upper(String name) {
        return name.toUpperCase(Locale.ROOT);
    }

    /** Tetrad's linear-Gaussian BIC under the missing-data policy the settings name. */
    static SemBicScore score(DataSet data, Settings s) {
        MissingDataSpec spec =
                MissingDataSpec.of(MissingDataPolicy.valueOf(upper(s.text("missingDataPolicy"))))
                        .withEmRidge(s.real("emRidge"))
                        .withEmTolerance(s.real("emTolerance"))
                        .withEmMaxIterations(s.integer("emMaxIterations"))
                        .withEssMode(MissingDataSpec.EffectiveSampleSizeMode.valueOf(
                                upper(s.text("essMode"))));
        SemBicScore score = new SemBicScore(data, s.bool("precomputeCovariances"), spec);
        score.setPenaltyDiscount(s.real("penaltyDiscount"));
        score.setStructurePrior(s.real("structurePrior"));
        score.setRuleType(SemBicScore.RuleType.valueOf(upper(s.text("ruleType"))));
        return score;
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
                String cell = cells[j].trim();
                rows[i - 1][j] = cell.equalsIgnoreCase("nan") ? Double.NaN
                        : Double.parseDouble(cell);
            }
        }
        return new BoxDataSet(new DoubleDataBox(rows), nodes);
    }
}
