package org.ruoyi.milvus;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.database.request.CreateDatabaseReq;
import io.milvus.v2.service.database.request.DescribeDatabaseReq;
import io.milvus.v2.service.database.response.DescribeDatabaseResp;
import io.milvus.v2.service.database.response.ListDatabasesResp;
import org.junit.Test;

import java.io.Console;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author jiajunjiang
 * @description Create by 2026/1/6 17:48
 */
public class MilvusDataBaseTest {
    @Test
    public void createdatabase() throws IOException {
        ConnectConfig config = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .token("root:Milvus")
                .build();
        MilvusClientV2 client = new MilvusClientV2(config);

        Map<String, String> properties = new HashMap<>();
        properties.put("database.replica.number", "3");
        CreateDatabaseReq createDatabaseReq = CreateDatabaseReq.builder()
                .databaseName("my_database_2")
                .properties(properties)
                .build();
        client.createDatabase(createDatabaseReq);

    }
    @Test
    public void queryDatabase() throws IOException {
        ConnectConfig config = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .token("root:Milvus")
                .build();
        MilvusClientV2 client = new MilvusClientV2(config);

        ListDatabasesResp listDatabasesResp = client.listDatabases();

        DescribeDatabaseResp descDBResp = client.describeDatabase(DescribeDatabaseReq.builder()
                .databaseName("my_database_2")
                .build());
        System.out.println(descDBResp);
    }
}
