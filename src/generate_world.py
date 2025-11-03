"""
Webots世界文件生成器

自动生成包含多个E-puck机器人的Webots世界文件
用于并行遗传算法训练

使用方法:
    python generate_world.py --robots 10 --output parallel_training.wbt
"""
import argparse
import os


def generate_world_file(num_robots=10, arena_size=3.0, track_radius=1.2, 
                       num_obstacles=5, output_file='parallel_training.wbt'):
    """
    生成Webots世界文件
    
    Args:
        num_robots: 机器人数量
        arena_size: 场地大小（米）
        track_radius: 轨道半径（米）
        num_obstacles: 障碍物数量
        output_file: 输出文件名
    """
    
    print(f"生成Webots世界文件...")
    print(f"  机器人数量: {num_robots}")
    print(f"  场地大小: {arena_size}m × {arena_size}m")
    print(f"  轨道半径: {track_radius}m")
    print(f"  障碍物数量: {num_obstacles}")
    
    # 世界文件头部
    wbt_content = f"""#VRML_SIM R2023b utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/gctronic/e-puck/protos/E-puck.proto"

WorldInfo {{
  info [
    "遗传算法训练世界"
    "包含{num_robots}个E-puck机器人用于并行训练"
    "环形轨道半径: {track_radius}m"
  ]
  title "GA并行训练场地"
  basicTimeStep 16
  contactProperties [
    ContactProperties {{
      material1 "wheel"
      material2 "floor"
      coulombFriction [
        0.5
      ]
    }}
  ]
}}

Viewpoint {{
  orientation -0.5773502691896258 0.5773502691896258 0.5773502691896258 2.0944
  position 0 0 {arena_size * 1.5}
  follow "supervisor"
}}

TexturedBackground {{
}}

TexturedBackgroundLight {{
}}

RectangleArena {{
  floorSize {arena_size} {arena_size}
  floorAppearance PBRAppearance {{
    baseColor 0.9 0.9 0.9
    roughness 1
    metalness 0
  }}
  wallHeight 0.2
}}

"""
    
    # 添加Supervisor机器人
    wbt_content += """# ==================== Supervisor机器人 ====================
# 这个机器人负责管理所有E-puck机器人的训练
Robot {
  translation 0 0 0.5
  name "supervisor"
  controller "parallel_trainer"
  supervisor TRUE
  children [
    Shape {
      appearance PBRAppearance {
        baseColor 0 0 1
        transparency 0.5
      }
      geometry Sphere {
        radius 0.05
      }
    }
  ]
}

"""
    
    # 计算机器人排列位置
    print(f"\n机器人位置:")
    robots_per_row = min(5, num_robots)  # 每行最多5个
    spacing = 0.3  # 机器人间距
    
    for i in range(num_robots):
        row = i // robots_per_row
        col = i % robots_per_row
        
        # 计算位置（居中排列）
        x = (col - (robots_per_row - 1) / 2.0) * spacing
        y = -track_radius - 0.5 - row * spacing  # 在轨道外侧
        z = 0.0
        
        print(f"  epuck_{i}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        wbt_content += f"""# E-puck机器人 {i}
E-puck {{
  translation {x:.3f} {y:.3f} {z:.3f}
  rotation 0 0 1 1.5708
  name "epuck_{i}"
  controller "<extern>"
  supervisor FALSE
  synchronization FALSE
  groundSensorsSlot [
    E-puckGroundSensors {{
    }}
  ]
}}

"""
    
    # 添加环形轨道（黑色圆环）
    wbt_content += f"""# ==================== 环形轨道 ====================
# 黑色圆环轨道
Solid {{
  translation 0 0 0.001
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0 0 0
        roughness 1
        metalness 0
      }}
      geometry Cylinder {{
        height 0.002
        radius {track_radius}
      }}
    }}
  ]
  name "track_outer"
}}

# 白色内圈（形成环形）
Solid {{
  translation 0 0 0.0015
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.9 0.9 0.9
        roughness 1
        metalness 0
      }}
      geometry Cylinder {{
        height 0.003
        radius {track_radius * 0.7}
      }}
    }}
  ]
  name "track_inner"
}}

"""
    
    # 添加障碍物
    print(f"\n障碍物位置:")
    import math
    for i in range(num_obstacles):
        # 在轨道上均匀分布障碍物
        angle = (2 * math.pi * i) / num_obstacles
        obstacle_radius = (track_radius + track_radius * 0.7) / 2  # 在轨道中间
        
        x = obstacle_radius * math.cos(angle)
        y = obstacle_radius * math.sin(angle)
        z = 0.05
        
        print(f"  obstacle_{i}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # 随机选择障碍物类型
        if i % 2 == 0:
            # 圆柱形障碍物
            wbt_content += f"""# 障碍物 {i} (圆柱)
Solid {{
  translation {x:.3f} {y:.3f} {z:.3f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 1 0 0
        roughness 1
        metalness 0
      }}
      geometry Cylinder {{
        height 0.1
        radius 0.05
      }}
    }}
  ]
  name "obstacle_{i}"
  boundingObject Cylinder {{
    height 0.1
    radius 0.05
  }}
  physics Physics {{
    density -1
    mass 1
  }}
}}

"""
        else:
            # 方形障碍物
            wbt_content += f"""# 障碍物 {i} (方形)
Solid {{
  translation {x:.3f} {y:.3f} {z:.3f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 1 0.5 0
        roughness 1
        metalness 0
      }}
      geometry Box {{
        size 0.08 0.08 0.1
      }}
    }}
  ]
  name "obstacle_{i}"
  boundingObject Box {{
    size 0.08 0.08 0.1
  }}
  physics Physics {{
    density -1
    mass 1
  }}
}}

"""
    
    # 添加起点标记
    wbt_content += f"""# ==================== 起点标记 ====================
Solid {{
  translation 0 {-track_radius} 0.002
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0 1 0
        roughness 1
        metalness 0
      }}
      geometry Box {{
        size 0.3 0.05 0.001
      }}
    }}
  ]
  name "start_line"
}}

"""
    
    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(wbt_content)
    
    print(f"\n✅ 世界文件已生成: {output_file}")
    print(f"\n使用方法:")
    print(f"1. 在Webots中打开: {output_file}")
    print(f"2. 确保parallel_trainer.py在controllers目录中")
    print(f"3. 点击播放按钮开始训练")
    
    return output_file


def generate_simple_world(num_robots=10, output_file='simple_training.wbt'):
    """
    生成简化版世界文件（用于快速测试）
    
    Args:
        num_robots: 机器人数量
        output_file: 输出文件名
    """
    
    print(f"生成简化版世界文件...")
    print(f"  机器人数量: {num_robots}")
    
    wbt_content = f"""#VRML_SIM R2023b utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/gctronic/e-puck/protos/E-puck.proto"

WorldInfo {{
  info [
    "简化训练世界"
  ]
  title "简化训练场地"
  basicTimeStep 16
}}

Viewpoint {{
  orientation -0.5 0.5 0.7 2.0
  position 0 0 3
}}

TexturedBackground {{
}}

TexturedBackgroundLight {{
}}

RectangleArena {{
  floorSize 2 2
}}

Robot {{
  name "supervisor"
  controller "parallel_trainer"
  supervisor TRUE
}}

"""
    
    # 添加机器人（一字排开）
    spacing = 0.3
    for i in range(num_robots):
        x = (i - num_robots / 2.0) * spacing
        
        wbt_content += f"""E-puck {{
  translation {x:.3f} 0 0
  name "epuck_{i}"
  controller "<extern>"
  groundSensorsSlot [
    E-puckGroundSensors {{
    }}
  ]
}}

"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(wbt_content)
    
    print(f"\n✅ 简化世界文件已生成: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成Webots并行训练世界文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成包含10个机器人的标准世界
  python generate_world.py --robots 10
  
  # 生成包含20个机器人的大型世界
  python generate_world.py --robots 20 --arena 5.0 --track 2.0
  
  # 生成简化版世界（快速测试）
  python generate_world.py --simple --robots 5
        """
    )
    
    parser.add_argument(
        '--robots', '-r',
        type=int,
        default=10,
        help='机器人数量（默认: 10）'
    )
    
    parser.add_argument(
        '--arena', '-a',
        type=float,
        default=3.0,
        help='场地大小（米，默认: 3.0）'
    )
    
    parser.add_argument(
        '--track', '-t',
        type=float,
        default=1.2,
        help='轨道半径（米，默认: 1.2）'
    )
    
    parser.add_argument(
        '--obstacles', '-o',
        type=int,
        default=5,
        help='障碍物数量（默认: 5）'
    )
    
    parser.add_argument(
        '--output', '-f',
        type=str,
        default='parallel_training.wbt',
        help='输出文件名（默认: parallel_training.wbt）'
    )
    
    parser.add_argument(
        '--simple', '-s',
        action='store_true',
        help='生成简化版世界（用于快速测试）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Webots世界文件生成器")
    print("=" * 60)
    print()
    
    if args.simple:
        generate_simple_world(args.robots, args.output)
    else:
        generate_world_file(
            num_robots=args.robots,
            arena_size=args.arena,
            track_radius=args.track,
            num_obstacles=args.obstacles,
            output_file=args.output
        )
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
