package org.ruoyi.milvus;

import com.google.gson.*;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.service.collection.request.*;
import io.milvus.v2.service.index.request.CreateIndexReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.QueryReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.BinaryVec;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.request.data.SparseFloatVec;
import io.milvus.v2.service.vector.response.InsertResp;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.database.request.CreateDatabaseReq;
import io.milvus.v2.service.database.request.DescribeDatabaseReq;
import io.milvus.v2.service.database.response.DescribeDatabaseResp;
import io.milvus.v2.service.database.response.ListDatabasesResp;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.response.InsertResp;
import io.milvus.v2.service.vector.response.QueryResp;
import io.milvus.v2.service.vector.response.SearchResp;
import org.junit.Test;

import java.io.IOException;
import java.util.*;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.DropCollectionReq;

/**
 * @author jiajunjiang
 * @description Create by 2026/1/6 17:48
 */
public class MilvusCollectionsTest {
    @Test
    public void createdatabase() throws IOException, InterruptedException {

        String CLUSTER_ENDPOINT = "http://localhost:19530";
        String TOKEN = "root:Milvus";

        // 1. Connect to Milvus server
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri(CLUSTER_ENDPOINT)
                .token(TOKEN)
                .build();

        MilvusClientV2 client = new MilvusClientV2(connectConfig);

        // 3. Create a collection in customized setup mode
        client.useDatabase("my_database_2");

        // 3.1 Create schema
        CreateCollectionReq.CollectionSchema schema = client.createSchema();

        // 3.2 Add fields to schema
        // 添加主字段
        schema.addField(AddFieldReq.builder()
                .fieldName("my_id")
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .autoID(false)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("my_vector")
                .dataType(DataType.FloatVector)//向量字段
                .dimension(5)//dim 参数表示向量字段中要保存的向量嵌入的维
                .build());
        schema.addField(AddFieldReq.builder()
                .fieldName("my_varchar")
                .dataType(DataType.VarChar)//标量字段
                .maxLength(512)
                .build());
        //添加数字字段
        schema.addField(AddFieldReq.builder()
                .fieldName("my_int64")
                .dataType(DataType.Int64)
                .build());


        //添加布尔字段
        schema.addField(AddFieldReq.builder()
                .fieldName("my_bool")
                .dataType(DataType.Bool)
                .build());

        // 添加json字段
        schema.addField(AddFieldReq.builder()
                .fieldName("my_json")
                .dataType(DataType.JSON)
                .build());

        // 3.3 Prepare index parameters
        IndexParam indexParamForIdField = IndexParam.builder()
                .fieldName("my_id")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .build();

        IndexParam indexParamForVectorField = IndexParam.builder()
                .fieldName("my_vector")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(IndexParam.MetricType.COSINE)
                .build();

        List<IndexParam> indexParams = new ArrayList<>();
        indexParams.add(indexParamForIdField);
        indexParams.add(indexParamForVectorField);

        // 3.4 Create a collection with schema and index parameters
        CreateCollectionReq customizedSetupReq1 = CreateCollectionReq.builder()
                .collectionName("customized_setup_1")
                .collectionSchema(schema)
                .indexParams(indexParams)
                .build();

        client.createCollection(customizedSetupReq1);

        // 3.5 Get load state of the collection
        GetLoadStateReq customSetupLoadStateReq1 = GetLoadStateReq.builder()
                .collectionName("customized_setup_1")
                .build();

        Boolean loaded = client.getLoadState(customSetupLoadStateReq1);
        System.out.println(loaded);


    }

    @Test
    public void autoID() throws IOException, InterruptedException {

        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build());
        client.useDatabase("my_database_2");
        CreateCollectionReq.CollectionSchema collectionSchema = CreateCollectionReq.CollectionSchema.builder()
                .build();
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("id")
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .autoID(true)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("embedding")
                .dataType(DataType.FloatVector)
                .dimension(4)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("category")
                .dataType(DataType.VarChar)
                .maxLength(1000)
                .build());


        // 3.3 Prepare index parameters
        IndexParam indexParamForIdField = IndexParam.builder()
                .fieldName("embedding")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .build();

        IndexParam indexParamForVectorField = IndexParam.builder()
                .fieldName("category")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(IndexParam.MetricType.COSINE)
                .build();

        List<IndexParam> indexParams = new ArrayList<>();
        indexParams.add(indexParamForIdField);
        indexParams.add(indexParamForVectorField);

        client.dropCollection(DropCollectionReq.builder()
                .collectionName("demo_autoid")
                .build());

        CreateCollectionReq requestCreate = CreateCollectionReq.builder()
                .collectionName("demo_autoid")
                .collectionSchema(collectionSchema).indexParams(indexParams)
                .build();
        client.createCollection(requestCreate);


        List<JsonObject> rows = new ArrayList<>();
        Gson gson = new Gson();
        JsonObject row1 = new JsonObject();
        row1.add("embedding", gson.toJsonTree(new float[]{0.1f, 0.2f, 0.3f, 0.4f}));
        row1.addProperty("category", "book");
        rows.add(row1);

        JsonObject row2 = new JsonObject();
        row2.add("embedding", gson.toJsonTree(new float[]{0.2f, 0.3f, 0.4f, 0.5f}));
        row2.addProperty("category", "toy");
        rows.add(row2);

        InsertResp insertR = client.insert(InsertReq.builder()
                .collectionName("demo_autoid")
                .data(rows)
                .build());
        System.out.printf("Generated IDs: %s\n", insertR.getPrimaryKeys());


        Thread.sleep(1000);


        // 6. Load the collection
        LoadCollectionReq loadCollectionReq = LoadCollectionReq.builder()
                .collectionName("demo_autoid")
                .build();

        client.loadCollection(loadCollectionReq);

        // 7. Get load state of the collection
        GetLoadStateReq loadStateReq = GetLoadStateReq.builder()
                .collectionName("demo_autoid")
                .build();

        Boolean res = client.getLoadState(loadStateReq);
        System.out.println(res);


    }

    // 使用密集向量
    @Test
    public void denseVector() throws Exception {


        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build());

        CreateCollectionReq.CollectionSchema schema = client.createSchema();
        schema.setEnableDynamicField(true);
        schema.addField(AddFieldReq.builder()
                .fieldName("pk")
                .dataType(DataType.VarChar)
                .isPrimaryKey(true)
                .autoID(true)
                .maxLength(100)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("dense_vector")
                .dataType(DataType.FloatVector) // 存储 32 位浮点数，常用于表示科学计算和机器学习中的实数。非常适合需要高精度的场景，例如区分相似向量。
                .dimension(4)
                .build());
        // 3.3 Prepare index parameters
        List<IndexParam> indexes = new ArrayList<>();

        indexes.add(IndexParam.builder()
                .fieldName("dense_vector")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(IndexParam.MetricType.IP)
                .build());

        // 3.4 Create collection
        client.dropCollection(DropCollectionReq.builder()
                .collectionName("my_collection")
                .build());
        CreateCollectionReq requestCreate = CreateCollectionReq.builder()
                .collectionName("my_collection")
                .collectionSchema(schema)
                .indexParams(indexes)
                .build();
        client.createCollection(requestCreate);
        // 3.5 Insert data
        List<JsonObject> rows = new ArrayList<>();
        Gson gson = new Gson();
        rows.add(gson.fromJson("{\"dense_vector\": [0.1, 0.2, 0.3, 0.4]}", JsonObject.class));
        rows.add(gson.fromJson("{\"dense_vector\": [0.2, 0.3, 0.4, 0.5]}", JsonObject.class));

        InsertResp insertR = client.insert(InsertReq.builder()
                .collectionName("my_collection")
                .data(rows)
                .build());
        //执行相似性搜索
        Map<String, Object> searchParams = new HashMap<>();
        searchParams.put("nprobe", 10);

        FloatVec queryVector = new FloatVec(new float[]{0.1f, 0.3f, 0.3f, 0.4f});

        SearchResp searchR = client.search(SearchReq.builder()
                .collectionName("my_collection")
                .data(Collections.singletonList(queryVector))
                .annsField("dense_vector")
                .searchParams(searchParams)
                .topK(5)
                .outputFields(Collections.singletonList("pk"))
                .build());

        System.out.println(searchR.getSearchResults());
    }

    /**
     * 使用二进制向量
     */
    @Test
    public void binaryVector() throws Exception {
        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build());

        CreateCollectionReq.CollectionSchema schema = client.createSchema();
        schema.setEnableDynamicField(true);
        schema.addField(AddFieldReq.builder()
                .fieldName("pk")
                .dataType(DataType.VarChar)
                .isPrimaryKey(true)
                .autoID(true)
                .maxLength(100)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("binary_vector")
                .dataType(DataType.BinaryVector)
                .dimension(16)
                .build());
        List<IndexParam> indexParams = new ArrayList<>();
        Map<String, Object> extraParams = new HashMap<>();

        indexParams.add(IndexParam.builder()
                .fieldName("binary_vector")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(IndexParam.MetricType.HAMMING)
                .build());


        // 3.4 Create collection
        client.dropCollection(DropCollectionReq.builder()
                .collectionName("my_collection")
                .build());
        CreateCollectionReq requestCreate = CreateCollectionReq.builder()
                .collectionName("my_collection")
                .collectionSchema(schema)
                .indexParams(indexParams)
                .build();
        client.createCollection(requestCreate);


        List<JsonObject> rows = new ArrayList<>();
        Gson gson = new Gson();
        {
            boolean[] boolArray = {true, false, false, true, true, false, true, true, false, true, false, false, true, true, false, true};
            JsonObject row = new JsonObject();
            row.add("binary_vector", gson.toJsonTree(convertBoolArrayToBytes(boolArray)));
            rows.add(row);
        }
        {
            boolean[] boolArray = {false, true, false, true, false, true, false, false, true, true, false, false, true, true, false, true};
            JsonObject row = new JsonObject();
            row.add("binary_vector", gson.toJsonTree(convertBoolArrayToBytes(boolArray)));
            rows.add(row);
        }

        InsertResp insertR = client.insert(InsertReq.builder()
                .collectionName("my_collection")
                .data(rows)
                .build());

        Map<String, Object> searchParams = new HashMap<>();
        searchParams.put("nprobe", 10);

        boolean[] boolArray = {true, false, false, true, true, false, true, true, false, true, false, false, true, true, false, true};
        BinaryVec queryVector = new BinaryVec(convertBoolArrayToBytes(boolArray));

        SearchResp searchR = client.search(SearchReq.builder()
                .collectionName("my_collection")
                .data(Collections.singletonList(queryVector))
                .annsField("binary_vector")
                .searchParams(searchParams)
                .topK(5)
                .outputFields(Collections.singletonList("pk"))
                .build());

        System.out.println(searchR.getSearchResults());
    }


    private static byte[] convertBoolArrayToBytes(boolean[] booleanArray) {
        byte[] byteArray = new byte[booleanArray.length / Byte.SIZE];
        for (int i = 0; i < booleanArray.length; i++) {
            if (booleanArray[i]) {
                int index = i / Byte.SIZE;
                int shift = i % Byte.SIZE;
                byteArray[index] |= (byte) (1 << shift);
            }
        }

        return byteArray;
    }

    // 稀疏向量
    @Test
    public void sparseVector() throws Exception {
        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build());

        //添加字段
        CreateCollectionReq.CollectionSchema schema = client.createSchema();
        schema.setEnableDynamicField(true);
        schema.addField(AddFieldReq.builder()
                .fieldName("pk")
                .dataType(DataType.VarChar)
                .isPrimaryKey(true)
                .autoID(true)
                .maxLength(100)
                .build());
        schema.addField(AddFieldReq.builder()
                .fieldName("sparse_vector")
                .dataType(DataType.SparseFloatVector)
                .build());
        schema.addField(AddFieldReq.builder()
                .fieldName("text")
                .dataType(DataType.VarChar)
                .maxLength(65535)
                .enableAnalyzer(true)
                .build());
//设置索引参数
        List<IndexParam> indexes = new ArrayList<>();

        Map<String, Object> extraParams = new HashMap<>();
        extraParams.put("inverted_index_algo", "DAAT_MAXSCORE"); // Algorithm used for building and querying the index

        indexes.add(IndexParam.builder()
                .fieldName("sparse_vector")
                .indexName("sparse_inverted_index")
                .indexType(IndexParam.IndexType.SPARSE_INVERTED_INDEX)
                .metricType(IndexParam.MetricType.IP)
                .extraParams(extraParams)
                .build());

        // 先drop collection
        client.dropCollection(DropCollectionReq.builder()
                .collectionName("my_collection")
                .build());
        CreateCollectionReq requestCreate = CreateCollectionReq.builder()
                .collectionName("my_collection")
                .collectionSchema(schema)
                .indexParams(indexes)
                .build();
        client.createCollection(requestCreate);

        //插入数据
        Gson gson = new Gson();
        List<JsonObject> rows = new ArrayList<>();

        {
            JsonObject row = new JsonObject();
            row.addProperty("text", "information retrieval is a field of study.");

            SortedMap<Long, Float> sparse = new TreeMap<>();
            sparse.put(1L, 0.5f);
            sparse.put(100L, 0.3f);
            sparse.put(500L, 0.8f);
            row.add("sparse_vector", gson.toJsonTree(sparse));
            rows.add(row);
        }
        {
            JsonObject row = new JsonObject();
            row.addProperty("text", "information retrieval focuses on finding relevant information in large datasets.");

            SortedMap<Long, Float> sparse = new TreeMap<>();
            sparse.put(10L, 0.1f);
            sparse.put(200L, 0.7f);
            sparse.put(1000L, 0.9f);
            row.add("sparse_vector", gson.toJsonTree(sparse));
            rows.add(row);
        }

        InsertResp insertResp = client.insert(InsertReq.builder()
                .collectionName("my_collection")
                .data(rows)
                .build());
        // Prepare search parameters
        Map<String, Object> searchParams = new HashMap<>();
        searchParams.put("drop_ratio_search", 0.2);

// Query with the sparse vector
        SortedMap<Long, Float> sparse = new TreeMap<>();
        sparse.put(1L, 0.2f);
        sparse.put(50L, 0.4f);
        sparse.put(1000L, 0.7f);
        SparseFloatVec queryData = new SparseFloatVec(sparse);

        SparseFloatVec queryVector = new SparseFloatVec(sparse);

        SearchResp searchR = client.search(SearchReq.builder()
                .collectionName("my_collection")
                .data(Collections.singletonList(queryData))
                .annsField("sparse_vector")
                .searchParams(searchParams)
                .consistencyLevel(ConsistencyLevel.STRONG)
                .topK(3)
                .outputFields(Collections.singletonList("pk"))
                .build());

        System.out.println(searchR.getSearchResults());

    }

    //字符串字段
    @Test
    public void stringField() throws Exception {
        //添加 VARCHAR 字段
        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build());

        CreateCollectionReq.CollectionSchema schema = client.createSchema();
        schema.setEnableDynamicField(true);

        schema.addField(AddFieldReq.builder()
                .fieldName("varchar_field1")
                .dataType(DataType.VarChar)
                .maxLength(100)
                .isNullable(true)
                .defaultValue("Unknown")
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("varchar_field2")
                .dataType(DataType.VarChar)
                .maxLength(200)
                .isNullable(true)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("pk")
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("embedding")
                .dataType(DataType.FloatVector)
                .dimension(3)
                .build());
        // 设置索引参数
        List<IndexParam> indexes = new ArrayList<>();
        indexes.add(IndexParam.builder()
                .fieldName("varchar_field1")
                .indexName("varchar_index")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .build());

        indexes.add(IndexParam.builder()
                .fieldName("embedding")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(IndexParam.MetricType.COSINE)
                .build());
        //drop
        DropCollectionReq requestDrop = DropCollectionReq.builder()
                .collectionName("my_collection")
                .build();
        client.dropCollection(requestDrop);
        // 创建 Collections
        CreateCollectionReq requestCreate = CreateCollectionReq.builder()
                .collectionName("my_collection")
                .collectionSchema(schema)
                .indexParams(indexes)
                .build();
        client.createCollection(requestCreate);

        //插入数据
        List<JsonObject> rows = new ArrayList<>();
        Gson gson = new Gson();
        rows.add(gson.fromJson("{\"varchar_field1\": \"Product A\", \"varchar_field2\": \"High quality product\", \"pk\": 1, \"embedding\": [0.1, 0.2, 0.3]}", JsonObject.class));
        rows.add(gson.fromJson("{\"varchar_field1\": \"Product B\", \"pk\": 2, \"embedding\": [0.4, 0.5, 0.6]}", JsonObject.class));
        rows.add(gson.fromJson("{\"varchar_field1\": null, \"varchar_field2\": null, \"pk\": 3, \"embedding\": [0.2, 0.3, 0.1]}", JsonObject.class));
        rows.add(gson.fromJson("{\"varchar_field1\": \"Product C\", \"varchar_field2\": null, \"pk\": 4, \"embedding\": [0.5, 0.7, 0.2]}", JsonObject.class));
        rows.add(gson.fromJson("{\"varchar_field1\": null, \"varchar_field2\": \"Exclusive deal\", \"pk\": 5, \"embedding\": [0.6, 0.4, 0.8]}", JsonObject.class));
        rows.add(gson.fromJson("{\"varchar_field1\": \"Unknown\", \"varchar_field2\": null, \"pk\": 6, \"embedding\": [0.8, 0.5, 0.3]}", JsonObject.class));
        rows.add(gson.fromJson("{\"varchar_field1\": \"\", \"varchar_field2\": \"Best seller\", \"pk\": 7, \"embedding\": [0.8, 0.5, 0.3]}", JsonObject.class));

        InsertResp insertR = client.insert(InsertReq.builder()
                .collectionName("my_collection")
                .data(rows)
                .build());

        String filter = "varchar_field1 == \"Product A\"";
        QueryResp resp = client.query(QueryReq.builder()
                .collectionName("my_collection")
                .filter(filter)
                .outputFields(Arrays.asList("varchar_field1", "varchar_field2"))
                .build());

        System.out.println(resp.getQueryResults());

        //使用过滤表达式进行向量搜索
        String filter2 = "varchar_field2 == \"Best seller\"";
        SearchResp resp2 = client.search(SearchReq.builder()
                .collectionName("my_collection")
                .annsField("embedding")
                .data(Collections.singletonList(new FloatVec(new float[]{0.3f, -0.6f, 0.1f})))
                .topK(5)
                .outputFields(Arrays.asList("varchar_field1", "varchar_field2"))
                .filter(filter)
                .build());

        System.out.println(resp2.getSearchResults());
    }
}
