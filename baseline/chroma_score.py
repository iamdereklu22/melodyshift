from ChromaCoverId.chroma_features import ChromaFeatures
import ChromaCoverId.cover_similarity_measures as sims

for i in range(1, 4):
    query_path = f'audios/benchmark-styletransfer-{i}.wav'   # now inside audios/
    reference_path = f'audios/original-audio-{i}.wav'        # now inside audios/

    chroma1 = ChromaFeatures(query_path)
    chroma2 = ChromaFeatures(reference_path)

    hpcp1 = chroma1.chroma_hpcp()
    hpcp2 = chroma2.chroma_hpcp()

    # Similarity matrix
    cross_recurrent_plot = sims.cross_recurrent_plot(hpcp1, hpcp2)

    # Cover song similarity distances
    qmax, cost_matrix_q = sims.qmax_measure(cross_recurrent_plot)
    dmax, cost_matrix_d = sims.dmax_measure(cross_recurrent_plot)

    # Nice printing
    print(f"--- Comparison {i} ---")
    print(f"Query file: {query_path}")
    print(f"Reference file: {reference_path}")
    print(f"Qmax distance: {qmax:.4f}")
    print(f"Dmax distance: {dmax:.4f}")
    print("----------------------\n")
