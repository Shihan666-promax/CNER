import os
import re
from collections import defaultdict, Counter
import time

class ACAutomaton:
    """AC自动机实现，用于高效的多模式字符串匹配"""
    
    def __init__(self):
        self.root = {}
        self.end_of_word = "#"
    
    def add_word(self, word):
        """添加词语到Trie树"""
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word
    
    def build_fail(self):
        """构建失败指针"""
        queue = []
        
        # 根节点的子节点的失败指针都指向根节点
        for key in self.root:
            if key != self.end_of_word:
                self.root[key]['fail'] = self.root
                queue.append(self.root[key])
        
        # 层次遍历构建失败指针
        while queue:
            current_node = queue.pop(0)
            
            for key in current_node:
                if key == self.end_of_word or key == 'fail':
                    continue
                
                # 子节点的失败指针
                child_node = current_node[key]
                fail_node = current_node['fail']
                
                # 如果失败指针节点有相同的子节点，则失败指针指向该子节点
                # 否则继续向上查找，直到根节点
                while fail_node != self.root and key not in fail_node:
                    fail_node = fail_node.get('fail', self.root)
                
                if key in fail_node:
                    child_node['fail'] = fail_node[key]
                else:
                    child_node['fail'] = self.root
                
                queue.append(child_node)
    
    def search(self, text):
        """在文本中搜索所有匹配的词语"""
        results = []
        current_node = self.root
        
        for i, char in enumerate(text):
            # 如果当前字符不在当前节点的子节点中，则通过失败指针跳转
            while current_node != self.root and char not in current_node:
                current_node = current_node.get('fail', self.root)
            
            if char in current_node:
                current_node = current_node[char]
            else:
                continue
            
            # 检查当前节点是否是某个词语的结尾
            temp = current_node
            while temp != self.root:
                if self.end_of_word in temp:
                    # 计算词语的起始位置
                    word_length = self._get_word_length(temp)
                    start_pos = i - word_length + 1
                    results.append((start_pos, i + 1, text[start_pos:i+1]))
                temp = temp.get('fail', self.root)
        
        return results
    
    def _get_word_length(self, node):
        """获取词语长度（辅助方法）"""
        length = 0
        while node != self.root and self.end_of_word not in node:
            node = node.get('fail', self.root)
            length += 1
        return length

class ChineseNER:
    """中文命名实体识别器"""
    
    def __init__(self, entity_dict_dir):
        self.entity_dict_dir = entity_dict_dir
        self.loc_entities = set()
        self.per_entities = set()
        self.org_entities = set()
        self.ac_automaton = ACAutomaton()
        self.entity_type_mapping = {}
        
        self.load_entity_dicts()
        self.build_ac_automaton()
    
    def load_entity_dicts(self):
        """加载实体词典"""
        print("正在加载实体词典...")
        
        # 加载地点实体词典
        loc_file = os.path.join(self.entity_dict_dir, "LOCdoc.txt")
        if os.path.exists(loc_file):
            with open(loc_file, 'r', encoding='utf-8') as f:
                self.loc_entities = set(line.strip() for line in f if line.strip())
            print(f"加载地点实体: {len(self.loc_entities)} 个")
        
        # 加载人名实体词典
        per_file = os.path.join(self.entity_dict_dir, "PERdoc.txt")
        if os.path.exists(per_file):
            with open(per_file, 'r', encoding='utf-8') as f:
                self.per_entities = set(line.strip() for line in f if line.strip())
            print(f"加载人名实体: {len(self.per_entities)} 个")
        
        # 加载组织机构实体词典
        org_file = os.path.join(self.entity_dict_dir, "ORGdoc.txt")
        if os.path.exists(org_file):
            with open(org_file, 'r', encoding='utf-8') as f:
                self.org_entities = set(line.strip() for line in f if line.strip())
            print(f"加载组织机构实体: {len(self.org_entities)} 个")
    
    def build_ac_automaton(self):
        """构建AC自动机"""
        print("正在构建AC自动机...")
        
        # 将所有实体添加到AC自动机，并记录实体类型
        for entity in self.loc_entities:
            self.ac_automaton.add_word(entity)
            self.entity_type_mapping[entity] = "LOC"
        
        for entity in self.per_entities:
            self.ac_automaton.add_word(entity)
            self.entity_type_mapping[entity] = "PER"
        
        for entity in self.org_entities:
            self.ac_automaton.add_word(entity)
            self.entity_type_mapping[entity] = "ORG"
        
        self.ac_automaton.build_fail()
        print("AC自动机构建完成")
    
    def recognize_entities(self, text):
        """识别文本中的命名实体"""
        # 使用AC自动机进行匹配
        matches = self.ac_automaton.search(text)
        
        # 处理匹配结果，处理重叠实体（优先选择较长的实体）
        matches.sort(key=lambda x: (x[0], -(x[1]-x[0])))  # 按起始位置排序，长度降序
        
        entities = []
        last_end = -1
        
        for start, end, entity_text in matches:
            if start >= last_end:  # 没有重叠
                entity_type = self.entity_type_mapping.get(entity_text, "UNK")
                entities.append({
                    'text': entity_text,
                    'start': start,
                    'end': end,
                    'type': entity_type
                })
                last_end = end
        
        return entities
    
    def text_to_bio(self, text, entities):
        """将文本和实体转换为BIO格式"""
        # 初始化所有token为O
        tokens = list(text)
        bio_tags = ['O'] * len(tokens)
        
        # 标记实体
        for entity in entities:
            start, end = entity['start'], entity['end']
            entity_type = entity['type']
            
            # 标记B标签
            if start < len(bio_tags):
                bio_tags[start] = f'B-{entity_type}'
            
            # 标记I标签
            for i in range(start + 1, min(end, len(bio_tags))):
                bio_tags[i] = f'I-{entity_type}'
        
        # 组合token和tag
        bio_result = []
        for token, tag in zip(tokens, bio_tags):
            bio_result.append(f"{token} {tag}")
        
        return bio_result

def load_original_text(file_path):
    """加载原始文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

def evaluate_ner(predictions, gold_standard):
    """评估NER性能"""
    # 这里需要根据你的标注数据格式来实现
    # 由于数据格式不明确，这里提供框架
    print("评估功能需要根据具体标注格式实现")
    return 0, 0, 0  # 返回准确率、召回率、F1

def analyze_entity_frequency(entities_list):
    """分析实体词频"""
    entity_counter = Counter()
    
    for entities in entities_list:
        for entity in entities:
            entity_type = entity['type']
            entity_text = entity['text']
            entity_counter[(entity_text, entity_type)] += 1
    
    # 按类型统计
    type_counter = defaultdict(int)
    for (text, entity_type), count in entity_counter.items():
        type_counter[entity_type] += count
    
    return entity_counter, type_counter

def main():
    """主函数"""
    # 配置路径
    base_dir = "data"
    entity_dict_dir = os.path.join(base_dir, "entitydocs")
    msra_dir = os.path.join(base_dir, "MSRA")
    
    original_text_file = os.path.join(msra_dir, "originaltext.txt")
    output_file = os.path.join(msra_dir, "ner_results.txt")
    bio_output_file = os.path.join(msra_dir, "ner_bio_results.txt")
    
    # 初始化NER系统
    print("初始化中文命名实体识别系统...")
    ner_system = ChineseNER(entity_dict_dir)
    
    # 加载原始文本
    print("加载原始文本...")
    sentences = load_original_text(original_text_file)
    print(f"共加载 {len(sentences)} 个句子")
    
    # 进行命名实体识别
    print("开始命名实体识别...")
    all_entities = []
    start_time = time.time()
    
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"处理第 {i} 个句子...")
        
        entities = ner_system.recognize_entities(sentence)
        all_entities.append(entities)
    
    end_time = time.time()
    print(f"实体识别完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 输出识别结果
    print("输出识别结果...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (sentence, entities) in enumerate(zip(sentences, all_entities)):
            f.write(f"句子 {i+1}: {sentence}\n")
            f.write("识别实体:\n")
            for entity in entities:
                f.write(f"  {entity['text']} - {entity['type']} (位置: {entity['start']}-{entity['end']})\n")
            f.write("-" * 50 + "\n")
    
    # 输出BIO格式结果
    print("输出BIO格式结果...")
    with open(bio_output_file, 'w', encoding='utf-8') as f:
        for sentence, entities in zip(sentences, all_entities):
            bio_tags = ner_system.text_to_bio(sentence, entities)
            for line in bio_tags:
                f.write(line + "\n")
            f.write("\n")
    
    # 实体词频统计
    print("进行实体词频统计...")
    entity_counter, type_counter = analyze_entity_frequency(all_entities)
    
    # 输出词频统计结果
    print("\n=== 实体词频统计结果 ===")
    print("\n按实体类型统计:")
    for entity_type, count in sorted(type_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{entity_type}: {count} 个")
    
    print(f"\n实体总数: {sum(type_counter.values())} 个")
    
    # 输出各类实体频次排序
    for entity_type in ['LOC', 'PER', 'ORG']:
        print(f"\n{entity_type}实体频次TOP 10:")
        type_entities = [(text, count) for (text, etype), count in entity_counter.items() 
                        if etype == entity_type]
        for text, count in sorted(type_entities, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {text}: {count} 次")
    
    # 总体实体频次排序
    print(f"\n总体实体频次TOP 20:")
    for (text, etype), count in entity_counter.most_common(20):
        print(f"  {text} [{etype}]: {count} 次")
    
    # 保存详细统计结果
    stats_file = os.path.join(msra_dir, "entity_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("实体词频详细统计\n")
        f.write("=" * 50 + "\n")
        
        f.write("\n按类型统计:\n")
        for entity_type, count in sorted(type_counter.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{entity_type}: {count} 个\n")
        
        f.write(f"\n实体总数: {sum(type_counter.values())} 个\n")
        
        for entity_type in ['LOC', 'PER', 'ORG']:
            f.write(f"\n{entity_type}实体频次排序:\n")
            type_entities = [(text, count) for (text, etype), count in entity_counter.items() 
                           if etype == entity_type]
            for text, count in sorted(type_entities, key=lambda x: x[1], reverse=True):
                f.write(f"  {text}: {count} 次\n")
    
    print(f"\n所有结果已保存到文件:")
    print(f"识别结果: {output_file}")
    print(f"BIO格式结果: {bio_output_file}")
    print(f"统计结果: {stats_file}")

if __name__ == "__main__":
    main()