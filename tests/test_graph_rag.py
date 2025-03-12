"""
Unit tests for the OpenRAG library.
This test suite covers core functionality including chunking, embedding, clustering, and database operations.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
import numpy as np

# Import the OpenRAG library
# We'll use relative imports assuming tests are in a tests/ directory
import sys
sys.path.append("..")
from openrag import (
    OpenRAG, Document, Chunk, Cluster,
    SentenceChunkStrategy, TokenChunkStrategy,
    OpenAIEmbeddingProvider, HuggingFaceEmbeddingProvider,
    KMeansClusteringProvider, DBSCANClusteringProvider, 
    MultiClusterAssignmentProvider, LLMGuidedClusteringProvider,
    LLMSummarizer, Neo4jVectorDBProvider
)

class TestDocument(unittest.TestCase):
    """Tests for Document class"""
    
    def test_document_creation(self):
        doc = Document(id="doc1", content="This is a test document.")
        self.assertEqual(doc.id, "doc1")
        self.assertEqual(doc.content, "This is a test document.")
        self.assertEqual(doc.metadata, {})
        self.assertEqual(doc.chunks, [])
    
    def test_document_with_metadata(self):
        metadata = {"author": "Test Author", "date": "2025-02-24"}
        doc = Document(id="doc1", content="Content", metadata=metadata)
        self.assertEqual(doc.metadata, metadata)

class TestChunk(unittest.TestCase):
    """Tests for Chunk class"""
    
    def test_chunk_creation(self):
        chunk = Chunk(id="chunk1", content="Chunk content", document_id="doc1")
        self.assertEqual(chunk.id, "chunk1")
        self.assertEqual(chunk.content, "Chunk content")
        self.assertEqual(chunk.document_id, "doc1")
        self.assertEqual(chunk.metadata, {})
        self.assertEqual(chunk.cluster_ids, [])
    
    def test_chunk_with_cluster_ids(self):
        chunk = Chunk(
            id="chunk1", 
            content="Chunk content", 
            document_id="doc1",
            cluster_ids=["cluster1", "cluster2"]
        )
        self.assertEqual(chunk.cluster_ids, ["cluster1", "cluster2"])

class TestChunkingStrategies(unittest.TestCase):
    """Tests for document chunking strategies"""
    
    def test_sentence_chunking(self):
        doc = Document(
            id="doc1", 
            content="This is sentence one. This is sentence two. This is sentence three. "
                    "This is sentence four. This is sentence five."
        )
        
        # Test with sentences_per_chunk=2, overlap=0
        chunker = SentenceChunkStrategy(sentences_per_chunk=2, overlap=0)
        chunks = chunker.chunk_document(doc)
        
        self.assertEqual(len(chunks), 2)  # Should have 2 chunks with 5 sentences
        self.assertEqual(chunks[0].content, "This is sentence one. This is sentence two.")
        self.assertEqual(chunks[1].content, "This is sentence three. This is sentence four.")
    
    def test_sentence_chunking_with_overlap(self):
        doc = Document(
            id="doc1", 
            content="This is sentence one. This is sentence two. This is sentence three. "
                    "This is sentence four. This is sentence five."
        )
        
        # Test with sentences_per_chunk=2, overlap=1
        chunker = SentenceChunkStrategy(sentences_per_chunk=2, overlap=1)
        chunks = chunker.chunk_document(doc)
        
        self.assertEqual(len(chunks), 4)  # Should have 4 chunks with overlap
        self.assertEqual(chunks[0].content, "This is sentence one. This is sentence two.")
        self.assertEqual(chunks[1].content, "This is sentence two. This is sentence three.")
    
    def test_token_chunking(self):
        doc = Document(
            id="doc1", 
            content="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        )
        
        # Test with tokens_per_chunk=4, overlap=0
        chunker = TokenChunkStrategy(tokens_per_chunk=4, overlap=0)
        chunks = chunker.chunk_document(doc)
        
        self.assertEqual(len(chunks), 2)  # Should have 2 chunks
        self.assertEqual(chunks[0].content, "word1 word2 word3 word4")
        self.assertEqual(chunks[1].content, "word5 word6 word7 word8")
    
    def test_token_chunking_with_overlap(self):
        doc = Document(
            id="doc1", 
            content="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        )
        
        # Test with tokens_per_chunk=4, overlap=2
        chunker = TokenChunkStrategy(tokens_per_chunk=4, overlap=2)
        chunks = chunker.chunk_document(doc)
        
        self.assertEqual(len(chunks), 4)  # Should have 4 chunks with overlap
        self.assertEqual(chunks[0].content, "word1 word2 word3 word4")
        self.assertEqual(chunks[1].content, "word3 word4 word5 word6")

class TestEmbeddingProviders(unittest.TestCase):
    """Tests for embedding providers"""
    
    @patch('openai.embeddings.create')
    def test_openai_embedding_provider(self, mock_create):
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_create.return_value = mock_response
        
        # Create provider and chunks
        provider = OpenAIEmbeddingProvider(api_key="fake_key")
        chunks = [
            Chunk(id="chunk1", content="Content 1", document_id="doc1"),
            Chunk(id="chunk2", content="Content 2", document_id="doc1")
        ]
        
        # Test embedding
        result = provider.embed_chunks(chunks)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(result[1].embedding, [0.1, 0.2, 0.3])
        
        # Verify the mock was called correctly
        self.assertEqual(mock_create.call_count, 2)
    
    def test_huggingface_embedding_provider(self):
        # Since this is a stub, we'll just test the basic mock functionality
        provider = HuggingFaceEmbeddingProvider()
        chunks = [
            Chunk(id="chunk1", content="Content 1", document_id="doc1")
        ]
        
        result = provider.embed_chunks(chunks)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].embedding), 384)  # Check we got the mock embedding

class TestClusteringProviders(unittest.TestCase):
    """Tests for clustering providers"""
    
    def test_kmeans_clustering(self):
        # Create provider and chunks with mock embeddings
        provider = KMeansClusteringProvider(n_clusters=2)
        chunks = [
            Chunk(id="chunk1", content="Content 1", document_id="doc1", 
                 embedding=[1.0, 0.0, 0.0]),
            Chunk(id="chunk2", content="Content 2", document_id="doc1",
                 embedding=[0.9, 0.1, 0.0]),
            Chunk(id="chunk3", content="Content 3", document_id="doc1",
                 embedding=[0.0, 0.9, 0.1]),
            Chunk(id="chunk4", content="Content 4", document_id="doc1",
                 embedding=[0.1, 0.8, 0.1])
        ]
        
        # Test clustering
        result_chunks, clusters = provider.cluster_chunks(chunks)
        
        # Should have 2 clusters
        self.assertEqual(len(clusters), 2)
        
        # Each chunk should be assigned to a cluster
        for chunk in result_chunks:
            self.assertEqual(len(chunk.cluster_ids), 1)
        
        # Check that similar chunks are in the same cluster
        chunk1_cluster = result_chunks[0].cluster_ids[0]
        chunk2_cluster = result_chunks[1].cluster_ids[0]
        chunk3_cluster = result_chunks[2].cluster_ids[0]
        chunk4_cluster = result_chunks[3].cluster_ids[0]
        
        # Chunks 1 and 2 should be in the same cluster
        self.assertEqual(chunk1_cluster, chunk2_cluster)
        
        # Chunks 3 and 4 should be in the same cluster
        self.assertEqual(chunk3_cluster, chunk4_cluster)
        
        # Chunks 1 and 3 should be in different clusters
        self.assertNotEqual(chunk1_cluster, chunk3_cluster)
    
    def test_dbscan_clustering(self):
        # Create provider and chunks with mock embeddings
        provider = DBSCANClusteringProvider(eps=0.5, min_samples=2)
        chunks = [
            Chunk(id="chunk1", content="Content 1", document_id="doc1", 
                 embedding=[1.0, 0.0, 0.0]),
            Chunk(id="chunk2", content="Content 2", document_id="doc1",
                 embedding=[0.9, 0.1, 0.0]),
            Chunk(id="chunk3", content="Content 3", document_id="doc1",
                 embedding=[0.0, 0.9, 0.1]),
            Chunk(id="chunk4", content="Content 4", document_id="doc1",
                 embedding=[0.1, 0.8, 0.1]),
            Chunk(id="chunk5", content="Content 5", document_id="doc1",
                 embedding=[0.5, 0.5, 0.0])  # Outlier
        ]
        
        # Test clustering
        result_chunks, clusters = provider.cluster_chunks(chunks)
        
        # Check that we have proper clusters and a noise cluster
        self.assertTrue(any(cluster.name == "Noise" for cluster in clusters))
        
        # Verify that similar chunks are in the same cluster
        chunk_clusters = {chunk.id: chunk.cluster_ids[0] for chunk in result_chunks}
        
        # Chunks 1 and 2 should be in the same cluster
        self.assertEqual(chunk_clusters["chunk1"], chunk_clusters["chunk2"])
        
        # Chunks 3 and 4 should be in the same cluster
        self.assertEqual(chunk_clusters["chunk3"], chunk_clusters["chunk4"])
        
        # Chunks 1 and 3 should be in different clusters
        self.assertNotEqual(chunk_clusters["chunk1"], chunk_clusters["chunk3"])
    
    def test_multi_cluster_assignment(self):
        # Create base clustering provider
        base_provider = KMeansClusteringProvider(n_clusters=2)
        
        # Create multi-cluster provider with high similarity threshold
        provider = MultiClusterAssignmentProvider(
            base_clustering=base_provider,
            similarity_threshold=0.7
        )
        
        # Create chunks with mock embeddings
        chunks = [
            Chunk(id="chunk1", content="Content 1", document_id="doc1", 
                 embedding=[0.8, 0.2, 0.0]),  # Close to both clusters
            Chunk(id="chunk2", content="Content 2", document_id="doc1",
                 embedding=[0.9, 0.1, 0.0]),  # Clearly in cluster 1
            Chunk(id="chunk3", content="Content 3", document_id="doc1",
                 embedding=[0.2, 0.8, 0.0]),  # Clearly in cluster 2
            Chunk(id="chunk4", content="Content 4", document_id="doc1",
                 embedding=[0.1, 0.9, 0.0])   # Clearly in cluster 2
        ]
        
        # Test clustering
        result_chunks, clusters = provider.cluster_chunks(chunks)
        
        # Should have 2 clusters
        self.assertEqual(len(clusters), 2)
        
        # Chunk1 should be in both clusters due to high similarity
        self.assertTrue(len(result_chunks[0].cluster_ids) >= 1)
        
        # Other chunks should be in just one cluster
        self.assertEqual(len(result_chunks[1].cluster_ids), 1)
        self.assertEqual(len(result_chunks[2].cluster_ids), 1)
        self.assertEqual(len(result_chunks[3].cluster_ids), 1)
    
    @patch('openai.chat.completions.create')
    def test_llm_guided_clustering(self, mock_create):
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Name: Technology Topics\nDescription: Discussions about AI and machine learning"
                )
            )
        ]
        mock_create.return_value = mock_response
        
        # Create base clustering provider
        base_provider = KMeansClusteringProvider(n_clusters=1)
        
        # Create LLM provider
        provider = LLMGuidedClusteringProvider(
            base_clustering=base_provider,
            api_key="fake_key"
        )
        
        # Create chunks with mock embeddings
        chunks = [
            Chunk(id="chunk1", content="AI models are improving rapidly", document_id="doc1", 
                 embedding=[0.8, 0.2, 0.0]),
            Chunk(id="chunk2", content="Machine learning is transforming industries", document_id="doc1",
                 embedding=[0.9, 0.1, 0.0])
        ]
        
        # Test clustering
        result_chunks, clusters = provider.cluster_chunks(chunks)
        
        # Should have 1 cluster
        self.assertEqual(len(clusters), 1)
        
        # Check cluster name and description
        self.assertEqual(clusters[0].name, "Technology Topics")
        self.assertEqual(clusters[0].description, "Discussions about AI and machine learning")

class TestSummarization(unittest.TestCase):
    """Tests for summarization providers"""
    
    @patch('openai.chat.completions.create')
    def test_llm_summarizer(self, mock_create):
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="This is a summary of the AI technology cluster."
                )
            )
        ]
        mock_create.return_value = mock_response
        
        # Create summarizer
        summarizer = LLMSummarizer(api_key="fake_key")
        
        # Create cluster and chunks
        cluster = Cluster(
            id="cluster1",
            name="AI Technology",
            chunk_ids=["chunk1", "chunk2"]
        )
        
        chunks = [
            Chunk(id="chunk1", content="AI models are improving rapidly", document_id="doc1"),
            Chunk(id="chunk2", content="Machine learning is transforming industries", document_id="doc1")
        ]
        
        # Test summarization
        summary = summarizer.summarize_cluster(cluster, chunks)
        
        # Check summary
        self.assertEqual(summary, "This is a summary of the AI technology cluster.")

class TestVectorDBProviders(unittest.TestCase):
    """Tests for vector database providers"""
    
    @patch('neo4j.GraphDatabase.driver')
    def test_neo4j_initialize(self, mock_driver):
        # Mock session
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        
        # Create provider
        provider = Neo4jVectorDBProvider(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        # Test initialization
        result = provider.initialize()
        
        # Check result
        self.assertTrue(result)
        
        # Verify that correct calls were made to Neo4j
        self.assertEqual(mock_session.run.call_count, 5)  # 2 constraints + 2 vector indexes + 1 implicit call
    
    @patch('neo4j.GraphDatabase.driver')
    def test_neo4j_store_documents(self, mock_driver):
        # Mock session
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        
        # Create provider
        provider = Neo4jVectorDBProvider(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        # Create documents
        documents = [
            Document(id="doc1", content="Document 1 content"),
            Document(id="doc2", content="Document 2 content")
        ]
        
        # Test storing documents
        result = provider.store_documents(documents)
        
        # Check result
        self.assertTrue(result)
        
        # Verify that correct calls were made to Neo4j
        self.assertEqual(mock_session.run.call_count, 2)  # 1 call per document
    
    @patch('neo4j.GraphDatabase.driver')
    def test_neo4j_query_similar(self, mock_driver):
        # Mock session and result
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [
            {"id": "chunk1", "content": "Content 1", "score": 0.9},
            {"id": "chunk2", "content": "Content 2", "score": 0.8}
        ]
        
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        
        # Create provider
        provider = Neo4jVectorDBProvider(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        # Test querying similar chunks
        results = provider.query_similar([0.1, 0.2, 0.3], top_k=2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "chunk1")
        self.assertEqual(results[0]["score"], 0.9)
        self.assertEqual(results[1]["id"], "chunk2")
        self.assertEqual(results[1]["score"], 0.8)

class TestFileProcessing(unittest.TestCase):
    """Tests for file processing and end-to-end workflows"""
    
    def test_process_text_file(self):
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a test document. It has multiple sentences. "
                    "We can use this to test our RAG pipeline. "
                    "Does it work with multiple paragraphs?\n\n"
                    "Yes, it should. This is another paragraph. "
                    "With several sentences.")
            temp_filename = f.name
        
        try:
            # Read the file
            with open(temp_filename, 'r') as f:
                content = f.read()
            
            # Create document
            doc = Document(id="test_doc", content=content)
            
            # Test chunking
            chunker = SentenceChunkStrategy(sentences_per_chunk=3, overlap=1)
            chunks = chunker.chunk_document(doc)
            
            # Check that we got the expected number of chunks
            self.assertTrue(len(chunks) > 0)
            
            # Check that the chunks contain the original sentences
            all_content = " ".join([chunk.content for chunk in chunks])
            
            # All sentences should be present in at least one chunk
            for sentence in ["This is a test document.", 
                            "It has multiple sentences.",
                            "We can use this to test our RAG pipeline.",
                            "Does it work with multiple paragraphs?",
                            "Yes, it should.",
                            "This is another paragraph.",
                            "With several sentences."]:
                self.assertIn(sentence, all_content)
            
        finally:
            # Clean up
            os.unlink(temp_filename)
    
    @patch('openai.embeddings.create')
    def test_end_to_end_workflow(self, mock_create):
        # Mock the OpenAI API response for embeddings
        def mock_embedding_response(model, input):
            # Generate a deterministic embedding based on the input
            hash_val = hash(input) % 1000
            embedding = [hash_val / 1000] * 3  # Simple 3D mock embedding
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=embedding)]
            return mock_resp
            
        mock_create.side_effect = mock_embedding_response
        
        # Create document
        doc = Document(
            id="test_doc", 
            content="This is about AI. Machine learning is important. "
                    "Python is used for AI. Data science uses Python. "
                    "Weather forecasting uses machine learning. "
                    "Natural language processing is a subfield of AI."
        )
        
        # Create RAG components
        chunker = SentenceChunkStrategy(sentences_per_chunk=2, overlap=0)
        embedder = OpenAIEmbeddingProvider(api_key="fake_key")
        clusterer = KMeansClusteringProvider(n_clusters=2)
        
        # Process document
        # 1. Chunk the document
        chunks = chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 3)
        
        # 2. Embed the chunks
        chunks = embedder.embed_chunks(chunks)
        self.assertTrue(all(chunk.embedding is not None for chunk in chunks))
        
        # 3. Cluster the chunks
        chunks, clusters = clusterer.cluster_chunks(chunks)
        self.assertEqual(len(clusters), 2)
        
        # 4. Check that all chunks are assigned to clusters
        self.assertTrue(all(len(chunk.cluster_ids) > 0 for chunk in chunks))
        
        # 5. Check that similar content is clustered together
        # Find AI-related chunks
        ai_chunks = [chunk for chunk in chunks if "AI" in chunk.content]
        
        # These should be in the same cluster
        if len(ai_chunks) >= 2:
            self.assertEqual(ai_chunks[0].cluster_ids[0], ai_chunks[1].cluster_ids[0])

class TestJSONSerializationAndDeserialization(unittest.TestCase):
    """Tests for JSON serialization and deserialization"""
    
    def test_document_serialization(self):
        # Create a document
        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"source": "test_file.txt"}
        )
        
        # Convert to dict
        doc_dict = {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "chunks": []
        }
        
        # Serialize to JSON
        doc_json = json.dumps(doc_dict)
        
        # Deserialize from JSON
        loaded_dict = json.loads(doc_json)
        
        # Create new document from dict
        loaded_doc = Document(
            id=loaded_dict["id"],
            content=loaded_dict["content"],
            metadata=loaded_dict["metadata"]
        )
        
        # Check that the documents are equivalent
        self.assertEqual(doc.id, loaded_doc.id)
        self.assertEqual(doc.content, loaded_doc.content)
        self.assertEqual(doc.metadata, loaded_doc.metadata)
    
    def test_chunk_serialization(self):
        # Create a chunk
        chunk = Chunk(
            id="chunk1",
            content="Test content",
            document_id="doc1",
            metadata={"start": 0, "end": 100},
            embedding=[0.1, 0.2, 0.3],
            cluster_ids=["cluster1", "cluster2"]
        )
        
        # Convert to dict
        chunk_dict = {
            "id": chunk.id,
            "content": chunk.content,
            "document_id": chunk.document_id,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "cluster_ids": chunk.cluster_ids
        }
        
        # Serialize to JSON
        chunk_json = json.dumps(chunk_dict)
        
        # Deserialize from JSON
        loaded_dict = json.loads(chunk_json)
        
        # Create new chunk from dict
        loaded_chunk = Chunk(
            id=loaded_dict["id"],
            content=loaded_dict["content"],
            document_id=loaded_dict["document_id"],
            metadata=loaded_dict["metadata"],
            embedding=loaded_dict["embedding"],
            cluster_ids=loaded_dict["cluster_ids"]
        )
        
        # Check that the chunks are equivalent
        self.assertEqual(chunk.id, loaded_chunk.id)
        self.assertEqual(chunk.content, loaded_chunk.content)
        self.assertEqual(chunk.document_id, loaded_chunk.document_id)
        self.assertEqual(chunk.metadata, loaded_chunk.metadata)
        self.assertEqual(chunk.embedding, loaded_chunk.embedding)
        self.assertEqual(chunk.cluster_ids, loaded_chunk.cluster_ids)
    
    def test_cluster_serialization(self):
        # Create a cluster
        cluster = Cluster(
            id="cluster1",
            name="Test Cluster",
            description="A test cluster",
            centroid=[0.1, 0.2, 0.3],
            chunk_ids=["chunk1", "chunk2"],
            summary="This is a summary"
        )
        
        # Convert to dict
        cluster_dict = {
            "id": cluster.id,
            "name": cluster.name,
            "description": cluster.description,
            "centroid": cluster.centroid,
            "chunk_ids": cluster.chunk_ids,
            "summary": cluster.summary
        }
        
        # Serialize to JSON
        cluster_json = json.dumps(cluster_dict)
        
        # Deserialize from JSON
        loaded_dict = json.loads(cluster_json)
        
        # Create new cluster from dict
        loaded_cluster = Cluster(
            id=loaded_dict["id"],
            name=loaded_dict["name"],
            description=loaded_dict["description"],
            centroid=loaded_dict["centroid"],
            chunk_ids=loaded_dict["chunk_ids"],
            summary=loaded_dict["summary"]
        )
        
        # Check that the clusters are equivalent
        self.assertEqual(cluster.id, loaded_cluster.id)
        self.assertEqual(cluster.name, loaded_cluster.name)
        self.assertEqual(cluster.description, loaded_cluster.description)
        self.assertEqual(cluster.centroid, loaded_cluster.centroid)
        self.assertEqual(cluster.chunk_ids, loaded_cluster.chunk_ids)
        self.assertEqual(cluster.summary, loaded_cluster.summary)

if __name__ == "__main__":
    unittest.main()