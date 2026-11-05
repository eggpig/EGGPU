// CPU-only sequential WebGraph export. Reflection keeps javac independent of downloaded jars.
import java.io.*;
import java.lang.reflect.Method;
import java.nio.file.*;

public final class LawToCsr {
    interface Rows { int n() throws Exception; int next() throws Exception; int[] neighbors() throws Exception; int degree() throws Exception; }
    static final class WebRows implements Rows {
        final Object graph, iterator;
        final Method n, next, neighbors, degree;
        WebRows(String basename) throws Exception {
            Class<?> g = Class.forName("it.unimi.dsi.webgraph.ImmutableGraph");
            Class<?> it = Class.forName("it.unimi.dsi.webgraph.NodeIterator");
            graph = g.getMethod("loadOffline", CharSequence.class).invoke(null, basename);
            iterator = g.getMethod("nodeIterator").invoke(graph);
            n = g.getMethod("numNodes"); next = it.getMethod("nextInt");
            neighbors = it.getMethod("successorArray"); degree = it.getMethod("outdegree");
        }
        public int n() throws Exception { return (Integer)n.invoke(graph); }
        public int next() throws Exception { return (Integer)next.invoke(iterator); }
        public int[] neighbors() throws Exception { return (int[])neighbors.invoke(iterator); }
        public int degree() throws Exception { return (Integer)degree.invoke(iterator); }
    }
    static void i32(OutputStream out, long value) throws IOException {
        if (value < 0 || value > Integer.MAX_VALUE) throw new IOException("signed int32 contract exceeded");
        for (int k = 0; k < 4; k++) out.write((int)(value >>> (8*k)) & 255);
    }
    static void export(Rows rows, Path out) throws Exception {
        int n = rows.n();
        if (n < 1) throw new IOException("empty vertex domain");
        Files.createDirectory(out);
        long raw = 0, loops = 0, duplicates = 0, arcs = 0;
        try (OutputStream offsets = new BufferedOutputStream(Files.newOutputStream(out.resolve("offsets.bin")), 1<<20);
             OutputStream indices = new BufferedOutputStream(Files.newOutputStream(out.resolve("indices.bin")), 1<<20)) {
            i32(offsets, 0);
            for (int u = 0; u < n; u++) {
                if (rows.next() != u) throw new IOException("unexpected node order");
                int d = rows.degree(); int[] targets = rows.neighbors(); int previous = -1;
                if (d < 0 || d > targets.length) throw new IOException("invalid outdegree");
                for (int j = 0; j < d; j++) {
                    int v = targets[j]; raw++;
                    if (v < 0 || v >= n || v < previous) throw new IOException("bad endpoint / unsorted row");
                    boolean duplicate = v == previous; previous = v;
                    if (v == u) { loops++; continue; }
                    if (duplicate) { duplicates++; continue; }
                    i32(indices, v); arcs++;
                }
                i32(offsets, arcs);
            }
        }
        String stats = String.format("{\"vertices\":%d,\"raw_arcs\":%d,\"arcs\":%d,\"self_loops_removed\":%d,\"duplicate_arcs_removed\":%d}%n", n, raw, arcs, loops, duplicates);
        Files.writeString(out.resolve("conversion.json"), stats);
        System.out.print(stats);
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("LawToCsr BASENAME|--self-test NEW_OUTPUT_DIRECTORY");
        Rows rows;
        if (args[0].equals("--self-test")) {
            rows = new Rows() {
                int u = -1; final int[][] x = {{0,1,1,3},{2},{},{0}, {}};
                public int n() { return x.length; } public int next() { return ++u; }
                public int[] neighbors() { return x[u]; } public int degree() { return x[u].length; }
            };
        } else rows = new WebRows(args[0]);
        export(rows, Path.of(args[1]));
    }
}
