# test_magic_pex.py
import asyncio
import argparse
from pathlib import Path
from app.tool.magic_pex import MagicPEXTool

async def test_magic_pex(gds_file, topcell, output_spice, output_dir):
    """Test the MagicPEXTool implementation"""
    
    # Check if GDS file exists
    if not Path(gds_file).exists():
        print(f"Error: GDS file not found: {gds_file}")
        print("\nSearching for GDS files in current directory:")
        gds_files = list(Path(".").glob("**/*.gds"))
        if gds_files:
            print("Available GDS files:")
            for f in gds_files[:10]:  # Show first 10
                print(f"  - {f}")
        else:
            print("  No GDS files found")
        return False
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing MagicPEXTool")
    print("=" * 60)
    print(f"GDS file:          {gds_file}")
    print(f"Top cell:          {topcell}")
    print(f"Output directory:  {output_dir}")
    print(f"Output SPICE:      {output_spice}")
    print("=" * 60)
    
    # Create the PEX tool
    pex_tool = MagicPEXTool()
    
    try:
        print("\nRunning PEX extraction...")
        result = await pex_tool.execute(
            gds_file=gds_file,
            topcell=topcell,
            output_spice=output_spice,
            output_dir=output_dir
        )
        
        print("\nPEX RESULTS:")
        print("=" * 60)
        if result.output:
            print(result.output)
        
        if result.error:
            print("\nERRORS:")
            print(result.error)
            return False
        
        # Check for PEX output file
        pex_file = Path(output_dir) / output_spice
        if not pex_file.exists():
            # Also check in current directory
            pex_file_alt = Path(output_spice)
            if pex_file_alt.exists():
                pex_file = pex_file_alt
        
        if pex_file.exists():
            print(f"\n✓ PEX netlist: {pex_file.absolute()} ({pex_file.stat().st_size} bytes)")
            
            # Show first few lines
            with open(pex_file, 'r') as f:
                lines = f.readlines()[:20]
            print("\nFirst 20 lines of PEX netlist:")
            for line in lines:
                print(f"  {line.rstrip()}")
            
            # Count parasitic elements
            with open(pex_file, 'r') as f:
                content = f.read()
            
            resistor_count = content.count(' R')
            capacitor_count = content.count(' C')
            
            print(f"\nParasitic Summary:")
            print(f"  Resistors:  {resistor_count}")
            print(f"  Capacitors: {capacitor_count}")
            
            return True
        else:
            print(f"\n✗ PEX netlist not found: {pex_file.absolute()}")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Test Magic PEX Tool for parasitic extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_magic_pex.py -g layout.gds -t MY_CELL -o pex.spice
  
  # With custom output directory
  python test_magic_pex.py -g layout.gds -t MY_CELL -o pex.spice -d ./pex_output/
  
  # Full example for OTA
  python test_magic_pex.py \\
    -g /path/to/ota.gds \\
    -t FIVE_TRANSISTOR_OTA_OPTIMAL \\
    -o ota_pex.spice \\
    -d ./pex_results/
        """
    )
    
    parser.add_argument(
        '-g', '--gds',
        type=str,
        required=True,
        help='Path to input GDS file'
    )
    
    parser.add_argument(
        '-t', '--topcell',
        type=str,
        required=True,
        help='Name of the top cell in the GDS file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='pex_output.spice',
        help='Path to output PEX SPICE file (default: pex_output.spice)'
    )
    
    parser.add_argument(
        '-d', '--output_dir',
        type=str,
        default='./layout/',
        help='Path to output directory for results (default: ./layout/)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=int,
        default=10,
        help='Resistance extraction tolerance (default: 10)'
    )
    
    parser.add_argument(
        '--cthresh',
        type=float,
        default=0.0,
        help='Capacitance threshold in fF (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    print("🧪 TESTING MAGIC PEX TOOL")
    print("=" * 60)
    
    # Run with user-provided arguments
    success = asyncio.run(test_magic_pex(
        gds_file=args.gds,
        topcell=args.topcell,
        output_spice=args.output,
        output_dir=args.output_dir
    ))
    
    if success:
        print("\n" + "="*60)
        print("✅ PEX EXTRACTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nResults saved to: {Path(args.output_dir).absolute()}")
        print(f"PEX netlist: {args.output}")
        print("\nNext steps:")
        print("  1. Use the PEX netlist for post-layout simulation")
        print("  2. Compare with pre-layout simulation results")
        print("  3. Analyze timing/performance with parasitics")
    else:
        print("\n" + "="*60)
        print("❌ PEX EXTRACTION FAILED")
        print("="*60)
        exit(1)

if __name__ == "__main__":
    main()