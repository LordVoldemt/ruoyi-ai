package org.ruoyi.service;

import org.ruoyi.common.core.exception.ServiceException;
import org.ruoyi.domain.bo.QueryVectorBo;
import org.ruoyi.domain.bo.StoreEmbeddingBo;

import java.util.List;

/**
 * 向量库管理服务接口
 *
 * <p>
 * 统一封装向量数据库（如 Milvus / PGVector / ES Vector 等）的核心操作，
 * 对上层屏蔽具体向量引擎实现差异。
 * </p>
 *
 * <p>
 * 典型使用场景：
 * <ul>
 *   <li>文档 / 知识切片 embedding 入库</li>
 *   <li>基于语义的向量相似度检索（RAG）</li>
 *   <li>按业务维度删除向量数据（docId / fid / id）</li>
 * </ul>
 * </p>
 *
 * <p>
 * 设计说明：
 * <ul>
 *   <li>接口层不暴露向量数据库细节（索引类型、距离度量等）</li>
 *   <li>支持多知识库（kid）+ 多 embedding 模型共存</li>
 * </ul>
 * </p>
 *
 * @author ageer
 */
public interface VectorStoreService {

    /**
     * 向量数据入库
     *
     * <p>
     * 将业务文本/文档切片对应的 embedding 向量写入向量库，
     * 通常包含向量本身 + 元数据（docId、fid、chunkId 等）。
     * </p>
     *
     * <p>
     * 典型流程：
     * <ol>
     *   <li>文本切分（chunk）</li>
     *   <li>调用 embedding 模型生成向量</li>
     *   <li>写入向量库并建立索引</li>
     * </ol>
     * </p>
     *
     * @param storeEmbeddingBo 向量入库参数对象，包含：
     *                         向量数据、业务标识、模型信息等
     * @throws ServiceException 向量入库失败时抛出（如连接异常、参数非法）
     */
    void storeEmbeddings(StoreEmbeddingBo storeEmbeddingBo) throws ServiceException;

    /**
     * 向量相似度查询
     *
     * <p>
     * 根据输入文本或向量，在指定知识库/模型下进行 TopK 相似度搜索，
     * 返回匹配的业务标识或文本内容。
     * </p>
     *
     * <p>
     * 常用于：
     * <ul>
     *   <li>RAG 场景中的语义检索</li>
     *   <li>知识库问答的上下文召回</li>
     * </ul>
     * </p>
     *
     * @param queryVectorBo 向量查询参数，通常包含：
     *                      查询向量 / 查询文本、TopK、kid、模型信息等
     * @return 相似度排序后的结果列表（通常为文本或业务主键）
     */
    List<String> getQueryVector(QueryVectorBo queryVectorBo);

    /**
     * 创建向量库 Schema / Collection
     *
     * <p>
     * 根据知识库 ID（kid）和 embedding 模型名称创建对应的向量表结构，
     * 通常在首次入库前调用。
     * </p>
     *
     * <p>
     * Schema 一般包含：
     * <ul>
     *   <li>向量字段（embedding）</li>
     *   <li>业务主键（id / chunkId）</li>
     *   <li>元数据字段（docId、fid 等）</li>
     * </ul>
     * </p>
     *
     * @param kid                知识库唯一标识
     * @param embeddingModelName embedding 模型名称（决定向量维度）
     */
    void createSchema(String kid, String embeddingModelName);

    /**
     * 按向量主键 ID 删除向量数据
     *
     * <p>
     * 用于精确删除单条向量记录，
     * 常见于数据修复或单条内容下架场景。
     * </p>
     *
     * @param id        向量数据唯一标识
     * @param modelName embedding 模型名称（定位具体向量表）
     * @throws ServiceException 删除失败时抛出
     */
    void removeById(String id, String modelName) throws ServiceException;

    /**
     * 按文档 ID 删除向量数据
     *
     * <p>
     * 删除某一整篇文档对应的所有向量切片，
     * 适用于文档整体删除或重新解析的场景。
     * </p>
     *
     * @param docId 文档唯一标识
     * @param kid   知识库 ID
     * @throws ServiceException 删除失败时抛出
     */
    void removeByDocId(String docId, String kid) throws ServiceException;

    /**
     * 按文件 ID 删除向量数据
     *
     * <p>
     * 通常用于批量删除同一文件来源的向量数据，
     * 与 docId 相比，fid 更偏向文件级别的业务抽象。
     * </p>
     *
     * @param fid 文件唯一标识
     * @param kid 知识库 ID
     * @throws ServiceException 删除失败时抛出
     */
    void removeByFid(String fid, String kid) throws ServiceException;
}
