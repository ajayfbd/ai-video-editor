print("Testing generator.py execution...")

try:
    # Read the file
    with open('ai_video_editor/modules/thumbnail_generation/generator.py', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Try to execute it in a clean namespace
    namespace = {}
    exec(content, namespace)
    
    # Check what was created
    classes = [name for name, obj in namespace.items() if isinstance(obj, type)]
    print(f"Classes found: {classes}")
    
    if 'ThumbnailGenerator' in classes:
        print("✓ ThumbnailGenerator found in namespace")
    else:
        print("✗ ThumbnailGenerator not found in namespace")
        print(f"All items in namespace: {list(namespace.keys())}")

except Exception as e:
    print(f"✗ Execution failed: {e}")
    import traceback
    traceback.print_exc()