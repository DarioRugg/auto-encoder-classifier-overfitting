for i in {1..3}; do
    # Add your bash commands here
    echo "command started"
    python test.py && wait
    # Wait for all background processes to finish
    echo "command finished"
done
