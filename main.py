"""
PCB透光画图层生成器 - v3.0
By ThatMrBen
"""

import cv2
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any


class PCBLightPaintingGenerator:
    def __init__(self, color_scheme: str = 'blue'):
        """初始化颜色方案"""
        self.color_scheme = color_scheme

        # 颜色定义（RGB格式）- 修正命名
        self.base_colors = {
            'black': [6, 16, 8],
            'white': [230, 234, 235],
            'dark_green': [25, 53, 34],  # 深绿色
            'yellow': [249, 225, 149],   # 黄色（前后都开窗，无铜皮）
        }

        # 色系定义
        self.color_schemes = {
            'blue': {
                'dark_main': [18, 47, 139],    # 深蓝
                'light_main': [153, 204, 249],  # 浅蓝
            },
            'red': {
                'dark_main': [125, 22, 22],    # 深红
                'light_main': [227, 93, 93],   # 浅红
            },
            'purple': {
                'dark_main': [125, 22, 125],   # 深紫
                'light_main': [227, 93, 227],  # 浅紫
            }
        }

        # 获取当前色系颜色
        self.colors = self.base_colors.copy()
        if color_scheme in self.color_schemes:
            self.colors.update(self.color_schemes[color_scheme])

        # 图层定义（使用英文key，显示中文名）
        self.layers = {
            'top_silkscreen': {
                'name_cn': '顶层丝印',
                'colors': ['white'],
                'description': '白色丝印区域'
            },
            'top_solder_mask': {
                'name_cn': '顶层阻焊',
                'colors': ['black', 'dark_green', 'yellow'],  # 修正：包含黄色
                'description': '黑色、深绿、黄色区域（顶层开窗区域）'
            },
            'top_copper': {
                'name_cn': '顶层铜皮',
                'colors': ['light_main', 'black'],
                'description': '浅色主色、黑色区域（顶层有铜皮区域）'
            },
            'bottom_solder_mask': {
                'name_cn': '背面阻焊',
                'colors': ['dark_main', 'light_main', 'black', 'yellow'],  # 修正：增加黄色
                'description': '深色主色、浅色主色、黑色、黄色区域（底层开窗区域）'
            },
            'bottom_copper': {
                'name_cn': '背面铜皮',
                'colors': ['dark_green'],
                'description': '深绿区域（底层有铜皮区域）'
            }
        }

        # 色系中文名
        self.scheme_names = {
            'blue': '蓝色系',
            'red': '红色系',
            'purple': '紫色系'
        }

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='PCB透光画图层生成器')
        parser.add_argument('input', help='输入图像路径')
        parser.add_argument('-o', '--output', help='输出目录')
        parser.add_argument('-c', '--color', choices=['blue', 'red', 'purple'],
                          default='blue', help='色系选择 (默认: blue)')
        parser.add_argument('-t', '--tolerance', type=int, default=10,
                          help='颜色匹配容差 (默认: 10)')
        parser.add_argument('-s', '--scale', type=float, default=1.0,
                          help='图像缩放倍数 (默认: 1.0)')
        parser.add_argument('--no-preview', action='store_true',
                          help='不生成预览图')

        return parser.parse_args()

    @staticmethod
    def load_and_preprocess_image(image_path: str, scale_factor: float = 1.0) -> np.ndarray:
        """加载和预处理图像"""
        print(f"加载图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 记录原始尺寸
        original_h, original_w = image.shape[:2]
        print(f"原始尺寸: {original_w}x{original_h}")

        # 缩放图像
        if scale_factor != 1.0:
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"缩放后尺寸: {new_w}x{new_h}")

        # 转换为RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def simplify_image_colors(self, image_rgb: np.ndarray, tolerance: int = 10) -> np.ndarray:
        """简化图像颜色到预定义颜色"""
        h, w = image_rgb.shape[:2]
        simplified = np.zeros_like(image_rgb)

        print(f"简化颜色... (容差: {tolerance})")

        # 预计算颜色距离表
        color_list = list(self.colors.values())

        for y in range(h):
            for x in range(w):
                pixel = image_rgb[y, x]
                best_idx = 0
                min_dist = float('inf')

                # 寻找最近的颜色
                for i, color in enumerate(color_list):
                    dist = np.sum((pixel - color) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                        if dist <= tolerance * tolerance:  # 平方距离
                            break

                simplified[y, x] = color_list[best_idx]

        return simplified

    def create_layer_mask(self, simplified_image: np.ndarray, color_names: List[str]) -> Tuple[np.ndarray, float]:
        """创建图层遮罩 - 透明底黑图"""
        h, w = simplified_image.shape[:2]
        mask = np.zeros((h, w, 4), dtype=np.uint8)  # 4通道：RGBA

        # 获取目标颜色
        target_colors = []
        for name in color_names:
            if name in self.colors:
                target_colors.append(self.colors[name])

        # 创建遮罩：目标颜色区域设为黑色不透明，其他区域完全透明
        for y in range(h):
            for x in range(w):
                pixel = simplified_image[y, x]
                for target in target_colors:
                    if np.array_equal(pixel, target):
                        # 黑色不透明：RGBA(0, 0, 0, 255)
                        mask[y, x] = [0, 0, 0, 255]
                        break

        # 计算黑色像素比例（用于统计）
        black_pixels = np.sum(np.all(mask[:, :, :3] == [0, 0, 0], axis=2) & (mask[:, :, 3] == 255))
        total_pixels = h * w
        black_ratio = (black_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return mask, black_ratio

    @staticmethod
    def add_alignment_marks(mask: np.ndarray) -> np.ndarray:
        """添加对齐标记到透明底黑图"""
        h, w = mask.shape[:2]
        marked = mask.copy()

        # 标记大小
        mark_size = max(1, min(h, w) // 200)

        # 在四角添加黑色对齐标记（不透明）
        # 左上
        marked[0:mark_size, 0:mark_size] = [0, 0, 0, 255]
        # 右上
        marked[0:mark_size, w-mark_size:w] = [0, 0, 0, 255]
        # 左下
        marked[h-mark_size:h, 0:mark_size] = [0, 0, 0, 255]
        # 右下
        marked[h-mark_size:h, w-mark_size:w] = [0, 0, 0, 255]

        return marked

    @staticmethod
    def save_transparent_image(image: np.ndarray, filepath: str) -> bool:
        """保存透明底黑图（RGBA格式）"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 确保是4通道RGBA图像
            if len(image.shape) == 2:
                # 单通道灰度图 -> 转为RGBA
                h, w = image.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[image == 255] = [0, 0, 0, 255]  # 白色区域转为黑色不透明
                image = rgba
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # RGB -> RGBA
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = image
                rgba[:, :, 3] = 255  # 完全不透明
                image = rgba

            # 保存为PNG（支持透明度）
            success = cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))

            return success

        except Exception as e:
            print(f"保存失败 {os.path.basename(filepath)}: {e}")
            return False

    @staticmethod
    def save_rgb_image(image: np.ndarray, filepath: str) -> bool:
        """保存普通RGB图像（不透明）"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if len(image.shape) == 2:
                # 单通道 -> 三通道
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # RGB -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            success = cv2.imwrite(filepath, image)
            return success

        except Exception as e:
            print(f"保存失败 {os.path.basename(filepath)}: {e}")
            return False

    def create_preview_image(self, simplified_image: np.ndarray) -> np.ndarray:
        """创建预览图像（黑色替换为银色，保持RGB）"""
        preview = simplified_image.copy()
        black_color = np.array(self.colors['black'])
        silver_color = np.array([160, 160, 160])  # 银色

        h, w = preview.shape[:2]
        for y in range(h):
            for x in range(w):
                if np.array_equal(preview[y, x], black_color):
                    preview[y, x] = silver_color

        return preview

    def generate_all_layers(self, simplified_image: np.ndarray, output_dir: str,
                           generate_preview: bool = True) -> List[Dict[str, Any]]:
        """生成所有图层 - 透明底黑图"""
        print(f"\n生成图层到目录: {output_dir}")
        print("-" * 60)

        results = []

        # 生成各图层（透明底黑图）
        for layer_key, layer_info in self.layers.items():
            cn_name = layer_info['name_cn']
            colors = layer_info['colors']

            print(f"处理: {cn_name}...", end=" ")
            mask, ratio = self.create_layer_mask(simplified_image, colors)
            marked_mask = self.add_alignment_marks(mask)

            # 保存文件（使用英文名）
            filename = f"{layer_key}.png"
            filepath = os.path.join(output_dir, filename)

            if self.save_transparent_image(marked_mask, filepath):
                print(f"✓ (黑色区域: {ratio:.1f}%)")
                results.append({
                    'key': layer_key,
                    'name_cn': cn_name,
                    'filename': filename,
                    'filepath': filepath,
                    'black_ratio': ratio,
                    'description': layer_info['description']
                })
            else:
                print("✗ 保存失败")

        # 保存简化后的图像（普通RGB，不透明）
        simplified_path = os.path.join(output_dir, "simplified.png")
        if self.save_rgb_image(simplified_image, simplified_path):
            print("✓ 保存: 简化后原图")
            results.append({
                'key': 'simplified',
                'name_cn': '简化后原图',
                'filename': 'simplified.png',
                'filepath': simplified_path,
                'description': '简化后的原始图像'
            })
        else:
            print("✗ 保存失败: 简化后原图")

        # 生成预览图（普通RGB，不透明）
        if generate_preview:
            preview = self.create_preview_image(simplified_image)
            preview_path = os.path.join(output_dir, "preview.png")
            if self.save_rgb_image(preview, preview_path):
                print("✓ 保存: 实物预览图")
                results.append({
                    'key': 'preview',
                    'name_cn': '实物预览图',
                    'filename': 'preview.png',
                    'filepath': preview_path,
                    'description': '模拟实物效果（黑色替换为银色）'
                })
            else:
                print("✗ 保存失败: 实物预览图")

        print("-" * 60)
        print(f"完成: {len([r for r in results if 'key' in r])} 个文件保存成功")

        return results

    def create_readme(self, output_dir: str, results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
        """创建说明文件（不带后缀）"""
        readme_path = os.path.join(output_dir, "README")

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("PCB透光画图层文件说明\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {args.input}\n")
            f.write(f"色系: {self.scheme_names.get(args.color, args.color)}\n")
            f.write(f"缩放倍数: {args.scale}\n")
            f.write(f"颜色容差: {args.tolerance}\n\n")

            f.write("文件列表:\n")
            f.write("-" * 60 + "\n")

            for result in results:
                if 'key' in result:
                    f.write(f"{result['filename']} - {result['name_cn']}\n")
                    f.write(f"  说明: {result['description']}\n")
                    if 'black_ratio' in result:
                        f.write(f"  黑色区域比例: {result['black_ratio']:.1f}%\n")
                    f.write("\n")

            f.write("\n文件格式说明:\n")
            f.write("-" * 60 + "\n")
            f.write("1. 所有图层文件（top_*.png, bottom_*.png）为透明底黑图\n")
            f.write("   - 黑色部分：有图形内容（铜皮、丝印或开窗区域）\n")
            f.write("   - 透明部分：无图形内容\n")
            f.write("   - 四角有黑色对齐标记，用于各层对齐\n")
            f.write("2. simplified.png 和 preview.png 为普通RGB图像，不透明\n\n")

            f.write("图层对应关系（用于PCB制版）:\n")
            f.write("-" * 60 + "\n")
            f.write("top_silkscreen.png   -> 顶层丝印层 (Top Silkscreen)\n")
            f.write("  格式：透明底黑图，黑色区域为丝印\n")
            f.write("top_solder_mask.png  -> 顶层阻焊层 (Top Solder Mask, 负片)\n")
            f.write("  格式：透明底黑图，黑色区域为开窗（无阻焊）\n")
            f.write("top_copper.png       -> 顶层铜皮层 (Top Copper)\n")
            f.write("  格式：透明底黑图，黑色区域为铜皮\n")
            f.write("bottom_solder_mask.png -> 底层阻焊层 (Bottom Solder Mask, 负片)\n")
            f.write("  格式：透明底黑图，黑色区域为开窗（无阻焊）\n")
            f.write("bottom_copper.png    -> 底层铜皮层 (Bottom Copper)\n")
            f.write("  格式：透明底黑图，黑色区域为铜皮\n")
            f.write("\n注意: 阻焊层文件为负片逻辑，黑色区域表示开窗（无阻焊）\n")

            # 颜色说明
            f.write("\n颜色含义:\n")
            f.write("-" * 60 + "\n")
            color_meanings = {
                'white': '顶层丝印',
                'black': '顶层开窗+有铜皮，底层开窗+无铜皮',
                'dark_green': '顶层开窗+无铜皮，底层不开窗+有铜皮',
                'yellow': '顶层开窗+无铜皮，底层开窗+无铜皮（透出PCB基板黄色）',
            }

            if self.color_scheme == 'blue':
                color_meanings.update({
                    'dark_main': '顶层不开窗+无铜皮，底层开窗+无铜皮（深蓝）',
                    'light_main': '顶层不开窗+有铜皮，底层开窗+无铜皮（浅蓝）'
                })
            elif self.color_scheme == 'red':
                color_meanings.update({
                    'dark_main': '顶层不开窗+无铜皮，底层开窗+无铜皮（深红）',
                    'light_main': '顶层不开窗+有铜皮，底层开窗+无铜皮（浅红）'
                })
            elif self.color_scheme == 'purple':
                color_meanings.update({
                    'dark_main': '顶层不开窗+无铜皮，底层开窗+无铜皮（深紫）',
                    'light_main': '顶层不开窗+有铜皮，底层开窗+无铜皮（浅紫）'
                })

            for color_key, meaning in color_meanings.items():
                color_value = self.colors.get(color_key, [0, 0, 0])
                f.write(f"{color_key}: RGB{tuple(color_value)} - {meaning}\n")

        print(f"✓ 创建说明文件: README")

    def run(self) -> int:
        """主运行函数"""
        print("PCB透光画图层生成器 v3.0")
        print("=" * 60)
        print("⚠️  注意：所有图层文件将生成为透明底黑图格式")
        print("   - 黑色部分：有图形内容")
        print("   - 透明部分：无图形内容")
        print("=" * 60)

        try:
            # 解析参数
            args = self.parse_arguments()
        except Exception as e:
            print(f"参数解析失败: {e}")
            return 1

        # 验证输入文件
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            return 1

        # 设置色系
        self.color_scheme = args.color
        if self.color_scheme in self.color_schemes:
            self.colors.update(self.color_schemes[self.color_scheme])

        print(f"色系: {self.scheme_names.get(self.color_scheme, self.color_scheme)}")

        # 创建输出目录
        if args.output:
            output_dir = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"pcb_output_{self.color_scheme}_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        try:
            # 加载和预处理图像
            image_rgb = self.load_and_preprocess_image(args.input, args.scale)

            # 简化颜色
            simplified = self.simplify_image_colors(image_rgb, args.tolerance)

            # 生成所有图层
            results = self.generate_all_layers(simplified, output_dir, not args.no_preview)

            # 创建说明文件
            self.create_readme(output_dir, results, args)

        except Exception as e:
            print(f"处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # 显示结果
        print(f"\n{'='*60}")
        print("✅ 处理完成!")
        print(f"输出目录: {os.path.abspath(output_dir)}")

        # 显示文件列表（使用英文名，但显示中文说明）
        print("\n📁 生成的文件:")
        print("-" * 60)

        total_size = 0
        successful_files = 0

        for result in results:
            if 'key' in result and os.path.exists(result['filepath']):
                size_kb = os.path.getsize(result['filepath']) / 1024
                total_size += size_kb
                successful_files += 1

                # 显示中文名和英文文件名
                print(f"  {result['filename']:25} ({result['name_cn']})")
                print(f"    {result['description']}")
                if 'black_ratio' in result:
                    print(f"    黑色区域: {result['black_ratio']:.1f}%")

                # 显示文件格式
                if result['key'] in ['simplified', 'preview']:
                    print(f"    格式: RGB图像（不透明）")
                else:
                    print(f"    格式: 透明底黑图（RGBA）")

                print(f"    文件大小: {size_kb:.1f} KB")
                print()

        print(f"总计: {successful_files} 个文件, 总大小: {total_size:.1f} KB")

        # 显示格式说明
        print("\n📋 文件格式说明:")
        print("-" * 60)
        print("🔲 透明底黑图（图层文件）:")
        print("  - top_silkscreen.png, top_solder_mask.png, top_copper.png")
        print("  - bottom_solder_mask.png, bottom_copper.png")
        print("  - 黑色区域：有图形内容")
        print("  - 透明区域：无图形内容")
        print("  - 四角有黑色对齐标记")
        print()
        print("🖼️ RGB图像（参考文件）:")
        print("  - simplified.png：简化后的原图")
        print("  - preview.png：实物预览图（黑色替换为银色）")

        # 显示使用建议
        print("\n🎯 使用建议:")
        print("1. 将生成的PNG文件上传到PCB制版网站（如嘉立创）")
        print("2. 透明底黑图格式符合PCB制版标准")
        print("3. 黑色区域将被识别为图形内容")
        print("4. 使用四角对齐标记确保各层对齐准确")

        return 0


def main() -> int:
    """程序入口"""
    try:
        generator = PCBLightPaintingGenerator()
        return generator.run()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断操作")
        return 130
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 尝试:")
        print("1. 检查输入文件路径是否正确")
        print("2. 检查是否有文件写入权限")
        print("3. 尝试不同的缩放倍数（如 -s 0.25）")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())