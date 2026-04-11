import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import '../../routes/app_routes.dart';
import '../chat/models/user_model.dart';
import '../dashboard_page/models/class_model.dart';

class PeoplePage extends StatefulWidget {
  final UserModel currentUser;
  final ClassModel classModel;

  const PeoplePage({
    Key? key,
    required this.currentUser,
    required this.classModel,
  }) : super(key: key);

  @override
  State<PeoplePage> createState() => _PeoplePageState();
}

class _PeoplePageState extends State<PeoplePage> {
  String _searchQuery = '';

  @override
  Widget build(BuildContext context) {
    final studentsQuery = FirebaseFirestore.instance
        .collection('users')
        .where('role', isEqualTo: 'Student')
        .where('joinedClassId', isEqualTo: widget.classModel.id)
        .snapshots();

    return Scaffold(
      appBar: AppBar(
        title: const Text('People'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          _buildSearchBar(context),
          Expanded(
            child: StreamBuilder<QuerySnapshot>(
              stream: studentsQuery,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }

                final docs = snapshot.data?.docs ?? [];

                if (docs.isEmpty) {
                  return const Center(child: Text("No students joined yet."));
                }

                final students = docs.map((doc) {
                  final data = doc.data() as Map<String, dynamic>;
                  return UserModel.fromMap(data);
                }).where((student) {
                  final query = _searchQuery.toLowerCase();
                  return student.name.toLowerCase().contains(query) ||
                      student.email.toLowerCase().contains(query);
                }).toList();

                if (students.isEmpty) {
                  return const Center(child: Text("No matching students."));
                }

                return ListView.builder(
                  itemCount: students.length,
                  itemBuilder: (context, index) {
                    final student = students[index];
                    return Card(
                      margin: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 6),
                      elevation: 2,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ListTile(
                        leading: CircleAvatar(
                          backgroundImage: student.photoUrl?.isNotEmpty == true
                              ? NetworkImage(student.photoUrl!)
                              : null,
                          child: student.photoUrl?.isEmpty == true ||
                                  student.photoUrl == null
                              ? Text(student.name.isNotEmpty
                                  ? student.name[0]
                                  : '?')
                              : null,
                        ),
                        title: Text(student.name),
                        subtitle: Text(student.email),
                        onTap: () {
                          Navigator.pushNamed(
                            context,
                            AppRoutes.studentInfoPage,
                            arguments: {
                              'user': student,
                              'classCode': widget.classModel.code,
                              'isReadOnly': true,
                            },
                          );
                        },
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(12.0),
      child: TextField(
        onChanged: (value) {
          setState(() => _searchQuery = value);
        },
        decoration: InputDecoration(
          hintText: 'Search students by name or email...',
          prefixIcon: const Icon(Icons.search),
          filled: true,
          fillColor: Colors.grey.shade100,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(20),
            borderSide: BorderSide.none,
          ),
        ),
      ),
    );
  }
}
