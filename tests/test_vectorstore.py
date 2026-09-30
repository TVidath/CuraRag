"""
Unit tests for Dense Vector Indexing & Embedding Engine using standard library unittest.
"""

import os
import sys

# Fix Windows PyTorch DLL loading
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    import site
    for site_pkg in site.getsitepackages():
        torch_lib = os.path.join(site_pkg, "torch", "lib")
        if os.path.exists(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass

import shutil
import tempfile
import unittest
from src.data.schema import ChildChunk
from src.vectorstore.vector_index import get_embedding_function, DenseVectorIndex


class TestDenseVectorIndex(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_chunks = [
            ChildChunk(
                chunk_id="mq_test_c0",
                parent_id="mq_test_p1",
                chunk_text="Glaucoma is a group of eye diseases that damage the optic nerve resulting in vision loss.",
                start_char=0,
                end_char=90,
                token_count=18,
                question="What is Glaucoma?",
                question_type="Definition / General Info",
                focus="Glaucoma",
                source="CDC"
            ),
            ChildChunk(
                chunk_id="mq_test_c1",
                parent_id="mq_test_p2",
                chunk_text="Symptoms of Urinary Tract Infections include burning sensation during urination and cloudy urine.",
                start_char=0,
                end_char=97,
                token_count=17,
                question="What are symptoms of UTI?",
                question_type="Symptoms & Signs",
                focus="Urinary Tract Infections",
                source="NIHSeniorHealth"
            )
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedding_function_dimensions_and_normalization(self):
        emb_fn = get_embedding_function()
        vec = emb_fn.embed_query("What is Glaucoma?")
        self.assertEqual(len(vec), 384)

    def test_build_and_query_dense_index(self):
        index_mgr = DenseVectorIndex(persist_directory=self.temp_dir, collection_name="test_collection")
        report = index_mgr.build_index(self.sample_chunks)
        
        self.assertEqual(report["indexed_count"], 2)
        self.assertEqual(report["dimensions"], 384)

        results = index_mgr.query_nearest_neighbors("eye disease optic nerve vision loss", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        
        self.assertEqual(top_match["child_id"], "mq_test_c0")
        self.assertEqual(top_match["parent_id"], "mq_test_p1")
        self.assertEqual(top_match["metadata"]["focus"], "Glaucoma")


if __name__ == "__main__":
    unittest.main()
